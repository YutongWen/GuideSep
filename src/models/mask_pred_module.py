from typing import Any, Optional
import os
import torch
import torchaudio
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
from torchmetrics import MeanMetric, MinMetric
from .utils import spec_fwd, spec_back
import torch.nn as nn
import statistics
import lightning as L
from torchmetrics.functional.audio.snr import signal_noise_ratio, scale_invariant_signal_noise_ratio
# from torchmetrics.functional.audio import signal_distortion_ratio
from torchmetrics.audio import SignalDistortionRatio
import torch.nn.functional as F
from itertools import zip_longest
import torchaudio.transforms as T
from torch.linalg import vector_norm
from einops import reduce

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def drop_mask(drop_prob, batch_size, device):
    _mask = torch.rand(batch_size)
    mask = (_mask < drop_prob)
    mask = mask.view(-1, 1, 1, 1).contiguous().to(device)

    return mask

def drop_mel_mask(p_drop_prob, n_drop_prob, batch_size, device):
    _p_mask = torch.rand(batch_size)
    _n_mask = torch.rand(batch_size)
    p_mask = (_p_mask < p_drop_prob)
    n_mask = (_n_mask < n_drop_prob)

    p_mask = p_mask.view(-1, 1, 1, 1).contiguous().to(device)
    n_mask = n_mask.view(-1, 1, 1, 1).contiguous().to(device)
    
    return p_mask, n_mask

def safe_signal_noise_ratio(preds: torch.Tensor, 
                            target: torch.Tensor, 
                            zero_mean: bool = False):

    return torch.nan_to_num(
        signal_noise_ratio(preds, target, zero_mean=zero_mean), nan=torch.nan, posinf=100.0, neginf=-100.0
    )


def safe_scale_invariant_signal_noise_ratio(preds: torch.Tensor, target: torch.Tensor):

    return torch.nan_to_num(
        scale_invariant_signal_noise_ratio(preds, target), nan=torch.nan, posinf=100.0,
        neginf=-100.0
    )

def safe_signal_distortion_ratio(preds: torch.Tensor, 
                            target: torch.Tensor,
                            sdr_metric):

    return torch.nan_to_num(
        sdr_metric(preds, target), nan=torch.nan, posinf=100.0, neginf=-100.0
    )


class SingleLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerMLP, self).__init__()
        
        # Define the hidden layer
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        
        # Define the activation function (ReLU in this case)
        self.activation = nn.ReLU()
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    # Define forward pass
    def forward(self, x):
        x = torch.transpose(x, -2, -1)
        # Pass input through hidden layer and activation function
        x = self.activation(self.hidden_layer(x))
        
        # Pass through the output layer
        x = self.output_layer(x)
        
        # swap the time and freq dimension to match the main model
        return torch.transpose(x, -2, -1)


class ConditionInsExtractionMaskPrediction(LightningModule):

    def __init__(
        self,
        condition_drop_prob: float,
        positive_mask_drop_prob: float,
        negative_mask_drop_prob: float,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        generated_frame_length: int,
        generated_frequency: int,
        use_ema: bool,
        ema_beta: float,
        ema_power: float,
        audio_sample_rate: int,
        hop_length: int,
        n_fft: int,
        center: bool = True,
        total_test_samples: Optional[int] = None,
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        multi_spectral_recon_loss_weight = 0.5,
        use_psuedo_masks = False
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.condition_drop_prob = condition_drop_prob
        self.positive_mask_drop_prob = positive_mask_drop_prob
        self.negative_mask_drop_prob = negative_mask_drop_prob
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, center=True)
        self.window = torch.hann_window(self.stft_args['n_fft'], periodic=True)
        self.use_psuedo_masks = use_psuedo_masks
        
        # diffusion components
        self.net = net
        self.use_ema = use_ema
        if self.use_ema:
            self.net_ema = EMA(self.net, beta=ema_beta, power=ema_power)
        # self.sampler = sampler
        # self.diffusion = diffusion
        # self.noise_distribution = noise_distribution # for training
        # self.noise_scheduler = noise_scheduler()     # for sampling
        self.generated_frame_length = generated_frame_length
        self.generated_frequency = generated_frequency
        
        # mask mapping
        self.mlp = SingleLayerMLP(input_size=80, hidden_size=256, output_size=256)
        
        # generation
        self.val_idx_rnd = [0]
        self.val_instances = []
        self.total_test_samples = total_test_samples
        self.audio_sample_rate = audio_sample_rate
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.sdr_metric = SignalDistortionRatio(load_diag=1e-6)
        
        # for multi-scale spectral loss
        self.mel_spec_transforms = nn.ModuleList([])
        self.mel_spec_recon_alphas = []
        
        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)
        
        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = (win_length / 2) ** 0.5

            calculated_n_fft = default(max(n_fft, win_length), win_length)  # @AndreyBocharnikov said this is usually win length, but overridable

            # if any audio experts have an opinion about these settings, please submit a PR

            melspec_transform = T.MelSpectrogram(
                sample_rate = audio_sample_rate,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = False
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)
        
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        
        # eval metric
        self.SiSDRi = []
        self.SiSDR = []
        self.SNR = []
        self.SDR = []
        self.SNR_ins = {}
        self.SiSDR_ins = {}
        self.SiSDRi_ins = {}
        self.SDR_ins = {}

        self.reference_saved = False
        
    def forward(self, clean, condition, mixture, residual):
        # Convert real and imaginary parts of x into two channel dimensions
        batch_size = clean.shape[0]
        cond_signal, mel_p_mask, mel_n_mask = condition
        mel_mask = torch.cat((mel_p_mask.unsqueeze(1), mel_n_mask.unsqueeze(1)), dim=1)
        mel_mask = self.mlp(mel_mask) # (b, 2, 256, 256)
        
        p_mask, n_mask = drop_mel_mask(p_drop_prob=self.positive_mask_drop_prob,
                                       n_drop_prob=self.negative_mask_drop_prob,
                                       batch_size=batch_size,
                                       device=self.device)
        mel_mask[:, :1, :, :] = torch.where(p_mask, torch.tensor(-1.0), mel_mask[:, :1, :, :])
        mel_mask[:, 1:, :, :] = torch.where(n_mask, torch.tensor(-1.0), mel_mask[:, 1:, :, :])
        
        spec = torch.stft(clean, window=self.window.to(self.device), normalized=True, 
                          return_complex=True, **self.stft_args)
        spec_mix = torch.stft(mixture, window=self.window.to(self.device), 
                               normalized=True, return_complex=True, 
                               **self.stft_args)
        spec_cond = torch.stft(cond_signal, window=self.window.to(self.device), 
                               normalized=True, return_complex=True, 
                               **self.stft_args)

        spec_mag = spec.abs().unsqueeze(1)
        spec_mix_mag = spec_mix.abs().unsqueeze(1)
        spec_cond_mag = spec_cond.abs().unsqueeze(1)
        
        # phase_gt = spec_mix.angle()
        
        condition_mask = drop_mask(drop_prob=self.condition_drop_prob, batch_size=batch_size, device=self.device)
        _no_mel_mask = p_mask & n_mask
        condition_mask = torch.where(_no_mel_mask, False, condition_mask)
        spec_cond_mag = torch.where(condition_mask, torch.tensor(-1.0), spec_cond_mag)
        
        # (batch, 256, 128)
        condition = torch.cat((spec_cond_mag, mel_mask), dim=1) # (batch, 5, 256, 256)
        # condition = condition * uncondition_masks
        mag_pred = self.net(x=spec_mix_mag, inj_channels=condition)
        mag_pred = mag_pred / torch.max(mag_pred + 1e-10)
        spec_mag = spec_mag / torch.max(spec_mag + 1e-10)
        
        # complex_spectrogram_pred = mag_pred.squeeze(1) * torch.exp(1j * phase_gt)
        
        # reconstructed_audio = torch.istft(
        #     complex_spectrogram_pred,
        #     window=self.window.to(self.device), 
        #     normalized=True,
        #     **self.stft_args            
        # )
        
        # compute loss, spectral mse + multi-scale spectral loss
        mse_losses = F.mse_loss(mag_pred, spec_mag)
        # mse_losses = reduce(mse_losses, "b ... -> b", "sum")
        # print(mse_losses)
        # multi_spectral_recon_loss = self.zero

        # if self.multi_spectral_recon_loss_weight > 0:
        #     for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
        #         orig_mel, recon_mel = map(mel_transform, (clean, reconstructed_audio))
        #         log_orig_mel, log_recon_mel = map(log, (orig_mel, recon_mel))
                
        #         nan_mask = torch.isnan(orig_mel)
        #         any_nan = torch.any(nan_mask)
        #         print("Any NaN in tensor?", any_nan.item())  # Output: True
                
        #         l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
        #         l2_log_mel_loss = alpha * vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

        #         print(l1_mel_loss, l2_log_mel_loss)
                
        #         multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss
                
        # print(multi_spectral_recon_loss) 
        # exit()
        
        # loss = mse_losses * (1 - self.multi_spectral_recon_loss_weight) + self.multi_spectral_recon_loss_weight * multi_spectral_recon_loss
        loss = mse_losses
        return loss

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass
        
    def on_train_epoch_start(self):
        # Get the current epoch number
        epoch_num = self.current_epoch + 150
        
        # Set the seed for the current epoch
        L.seed_everything(42 + epoch_num)  # Set a unique seed per epoch
        print(f"Epoch {epoch_num} uses seed {42 + epoch_num}")

    def model_step(self, batch: Any):
        clean, cond_signal, mel_p_mask, mel_n_mask, mixture, residual, _, _, _, _ = batch

        loss = self.forward(clean=clean, 
                            condition=(cond_signal, mel_p_mask, mel_n_mask), 
                            mixture=mixture,
                            residual=residual)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        
        if self.use_ema:
            # Update EMA model and log decay
            self.net_ema.update()
            self.log("ema_decay", self.net_ema.get_current_decay())
        return {"loss": loss}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass
    
    def on_validation_epoch_start(self):
        L.seed_everything(42)
    
    def validation_step(self, batch: Any, batch_idx: int):

        loss_mean = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss_mean)
        self.log("val/loss", self.val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx in self.val_idx_rnd and len(self.val_instances) < len(self.val_idx_rnd):

            # clean, midi, mixture, residual, tar_ins = batch
            # clean, cond_signal, mixture, residual, tar_ins, cond_ins_name = batch
            clean, cond_signal, mel_p_mask, mel_n_mask, mixture, residual, tar_ins, cond_ins_name, mel_spec_tar, mel_spec_res = batch

            self.val_instances.append((clean[batch_idx].unsqueeze(0), 
                                       cond_signal[batch_idx].unsqueeze(0),
                                       mel_p_mask[batch_idx].unsqueeze(0),
                                       mel_n_mask[batch_idx].unsqueeze(0),
                                       mixture[batch_idx].unsqueeze(0),
                                       residual[batch_idx].unsqueeze(0),
                                       tar_ins[batch_idx],
                                       cond_ins_name[batch_idx],
                                       mel_spec_tar[batch_idx].unsqueeze(0), 
                                       mel_spec_res[batch_idx].unsqueeze(0)))
        
        return {"loss": loss_mean}
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        diff_net = self.net_ema if self.use_ema else self.net
        for i, val_instance in enumerate(self.val_instances):

            # clean, midi, mixture, midi_others, audio_others, tar_ins = val_instance
            clean, cond_signal, mel_p_mask, mel_n_mask, mixture, residual, tar_ins, cond_ins_name, mel_spec_tar, mel_spec_res = val_instance
            
            # save restored audio sample
            audio_path = os.path.join(audio_save_dir, f'val_{i}_{tar_ins}_{self.device.index}_{self.global_step}.wav')
            
            p_mask, n_mask = drop_mel_mask(p_drop_prob=1.0,
                                        n_drop_prob=1.0,
                                        batch_size=clean.shape[0],
                                        device=self.device)
            
            with torch.no_grad():
                audio_sample = self.inference(condition=(cond_signal, mel_p_mask, mel_n_mask), 
                                                        mixture=mixture, 
                                                        diff_net=diff_net,
                                                        p_mask=p_mask,
                                                        n_mask=n_mask)
            audio_sample = audio_sample / torch.max(torch.abs(audio_sample + 1e-10))
            torchaudio.save(audio_path, audio_sample, self.audio_sample_rate) 
            
            if not self.reference_saved:
                mixture_ref_path = os.path.join(audio_save_dir, f'val_mixture_{i}_{tar_ins}_{self.device.index}.wav')
                clean_ref_path = os.path.join(audio_save_dir, f'val_reference_{i}_{tar_ins}_{self.device.index}.wav')
                residual_ref_path = os.path.join(audio_save_dir, f'val_reference_{i}_{tar_ins}_{self.device.index}_res.wav')
                cond_ref_path = os.path.join(audio_save_dir, f'val_cond_{i}_{tar_ins}_{self.device.index}_{cond_ins_name}.wav')
                mel_mask_ref_path = os.path.join(audio_save_dir, f'val_mask_{i}_{tar_ins}_{self.device.index}.pdf')
                
                # save clean reference audio
                torchaudio.save(clean_ref_path, clean.cpu(), self.audio_sample_rate)
                torchaudio.save(residual_ref_path, residual.cpu(), self.audio_sample_rate)
                
                # save mixture reference speech
                torchaudio.save(mixture_ref_path, mixture.cpu(), self.audio_sample_rate)
                torchaudio.save(cond_ref_path, cond_signal.cpu(), self.audio_sample_rate)
                
                # save plot
                self.plot_mel_mask(mel_spec_mask_tar=mel_p_mask,
                                   mel_spec_mask_res=mel_n_mask,
                                   mel_spec_tar=mel_spec_tar,
                                   mel_spec_res=mel_spec_res,
                                   save_path=mel_mask_ref_path)

        self.reference_saved = True
        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True, sync_dist=True)
        torch.cuda.empty_cache()
        
    def plot_mel_mask(self, mel_spec_mask_tar, mel_spec_mask_res, mel_spec_tar, mel_spec_res, save_path):
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        im1 = axs[0, 0].imshow(mel_spec_mask_tar[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[0, 0].set_title('Mel-mask target')
        fig.colorbar(im1, ax=axs[0, 0])

        im2 = axs[0, 1].imshow(mel_spec_mask_res[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[0, 1].set_title('Mel-mask res')
        fig.colorbar(im2, ax=axs[0, 1])

        im3 = axs[1, 0].imshow(mel_spec_tar[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[1, 0].set_title('Mel-Spectrogram target')
        fig.colorbar(im3, ax=axs[1, 0])

        im4 = axs[1, 1].imshow(mel_spec_res[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[1, 1].set_title('Mel-Spectrogram res')
        fig.colorbar(im4, ax=axs[1, 1])
        
        for ax in axs.flat:
            ax.set(xlabel='Time', ylabel='Mel Frequency')

        # Show the plots
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)

    @torch.no_grad()
    def inference(self, condition=None, 
                  mixture=None, 
                  diff_net=None, 
                  p_mask=None, 
                  n_mask=None):
        with torch.no_grad():
            # input mixture
            condition, mel_p_mask, mel_n_mask = condition
            if condition is None:
                spec_cond_mag = -1.0 * torch.ones((mixture.shape[0], 1, self.n_fft//2+1, self.generated_frame_length), device=self.device)
            else:
                condition = torch.stft(condition, window=self.window.to(self.device), 
                                normalized=True, return_complex=True, 
                                **self.stft_args)
                spec_cond_mag = condition.abs().unsqueeze(1)
        
            mel_mask = torch.cat((mel_p_mask.unsqueeze(1), mel_n_mask.unsqueeze(1)), dim=1)
            mel_mask = self.mlp(mel_mask)
            if p_mask is not None and n_mask is not None:
                mel_mask[:, :1, :, :] = torch.where(p_mask, torch.tensor(-1.0), mel_mask[:, :1, :, :])
                mel_mask[:, 1:, :, :] = torch.where(n_mask, torch.tensor(-1.0), mel_mask[:, 1:, :, :])
        
               
            # separation
            spec_mixture = torch.stft(mixture, window=self.window.to(self.device), normalized=True, 
                                return_complex=True, **self.stft_args)
            spec_mix_mag = spec_mixture.abs().unsqueeze(1)
            phase_gt = spec_mixture.angle()
            
            condition = torch.cat((spec_cond_mag, mel_mask), dim=1) # (batch, 5, 256, 256)
            
            mag_pred = diff_net(x=spec_mix_mag, inj_channels=condition)
        
            complex_spectrogram_pred = mag_pred.squeeze(1) * torch.exp(1j * phase_gt)
            
            reconstructed_audio = torch.istft(
                complex_spectrogram_pred,
                window=self.window.to(self.device), 
                normalized=True,
                **self.stft_args            
            )
            
        return reconstructed_audio.cpu()

    def test_step(self, batch: Any, batch_idx: int):
        L.seed_everything(42)
        # loss = self.model_step(batch)
        
        # # update and log metrics
        # self.val_loss(loss)
        # self.log("val/loss", self.val_loss, on_step=False, 
        #          on_epoch=True, prog_bar=True, sync_dist=True)

        # clean, midi, mixture, midi_others, audio_others, tar_ins = batch
        # clean, midi, mixture, residual, tar_ins = batch
        clean, cond_signal, mel_p_mask, mel_n_mask, mixture, _, tar_ins, cond_ins_name, _, _, mel_spec_mask_cond, mel_spec_mask_mix = batch
        if clean is None:
            # skip this test batch
            print('skip this test batch')
            return 0
        # mixture, cond_signal, mel_p_mask, mel_n_mask = batch
        diff_net = self.net_ema if self.use_ema else self.net
        if self.positive_mask_drop_prob != 0 and self.negative_mask_drop_prob != 0:
            p_mask, n_mask = drop_mel_mask(p_drop_prob=self.positive_mask_drop_prob,
                                        n_drop_prob=self.negative_mask_drop_prob,
                                        batch_size=cond_signal.shape[0],
                                        device=self.device)
        else:
            p_mask = None
            n_mask = None
            
        if self.condition_drop_prob == 1:
            print('drop melody condition')
            cond_signal = None
            
        if p_mask is None:
            print('positive mask is not using')
            
        if n_mask is None:
            print('negative mask is not using')
            
        if self.use_psuedo_masks:
            print('use psuedo masks')
            mel_p_mask = mel_spec_mask_cond
            mel_n_mask = mel_spec_mask_mix
        
        
        with torch.no_grad():
            audio_sample = self.inference(condition=(cond_signal, mel_p_mask, mel_n_mask), 
                                          mixture=mixture, 
                                          diff_net=diff_net,
                                          p_mask=p_mask,
                                          n_mask=n_mask)
            
        audio_sample = audio_sample / torch.max(torch.abs(audio_sample + 1e-10))
        # compute SiSDRi
        sample_sisdr = safe_scale_invariant_signal_noise_ratio(audio_sample, clean.cpu()).item()
        sample_sisdri = (sample_sisdr - safe_scale_invariant_signal_noise_ratio(mixture.cpu(), clean.cpu())).item()
        sample_snr = safe_signal_noise_ratio(audio_sample, clean.cpu()).item()
        sample_sdr = safe_signal_distortion_ratio(audio_sample, clean.cpu(), sdr_metric=self.sdr_metric).item()
        self.SiSDRi.append(sample_sisdri)
        self.SiSDR.append(sample_sisdr)
        self.SNR.append(sample_snr)
        self.SDR.append(sample_sdr)
        
        ins_key = tar_ins[0].split('_')[0]
        if ins_key not in self.SiSDRi_ins.keys():
            self.SiSDRi_ins[ins_key] = [sample_sisdri]
            self.SiSDR_ins[ins_key] = [sample_sisdr]
            self.SNR_ins[ins_key] = [sample_snr]
            self.SDR_ins[ins_key] = [sample_sdr]
        else:
            self.SiSDRi_ins[ins_key].append(sample_sisdri)
            self.SiSDR_ins[ins_key].append(sample_sisdr)
            self.SNR_ins[ins_key].append(sample_snr)
            self.SDR_ins[ins_key].append(sample_sdr)
            
        # tar_ins = ['']
        # cond_ins_name = ['']
        # sample_sisdri = 0
        # save restored audio sample
        if batch_idx % 100 == 0:
            audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
            os.makedirs(audio_save_dir, exist_ok=True)
            tar_audio_path = os.path.join(audio_save_dir, 'val_' + str(batch_idx) + '_' + tar_ins[0] + '_' + str(sample_sisdri) + '.wav')
            torchaudio.save(tar_audio_path, audio_sample, self.audio_sample_rate) 
            
            # save mixture and clean reference and midi
            clean_ref_path = os.path.join(audio_save_dir, 'val_' + str(batch_idx) + '_clean_ref.wav')
            torchaudio.save(clean_ref_path, clean.cpu(), self.audio_sample_rate)
            mixture_ref_path = os.path.join(audio_save_dir, 'val_' + str(batch_idx) + '_mixture_ref.wav')
            torchaudio.save(mixture_ref_path, mixture.cpu(), self.audio_sample_rate)
        
            if cond_signal is not None:
                cond_ref_path = os.path.join(audio_save_dir, 'val_' + str(batch_idx) + '_cond_' + cond_ins_name[0] + '_ref.wav')
                torchaudio.save(cond_ref_path, cond_signal.cpu(), self.audio_sample_rate)
            mask_ref_path = os.path.join(audio_save_dir, 'val_' + str(batch_idx) + '_mask_ref.pdf')
            # self.plot_mel_mask(mel_spec_mask_tar=mel_p_mask,
            #                    mel_spec_mask_res=mel_n_mask,
            #                    mel_spec_tar=mel_spec_tar,
            #                    mel_spec_res=mel_spec_res,
            #                    save_path=mask_ref_path)
            self.plot_mel_mask(mel_spec_mask_tar=mel_p_mask,
                            mel_spec_mask_res=mel_n_mask,
                            mel_spec_tar=mel_p_mask,
                            mel_spec_res=mel_n_mask,
                            save_path=mask_ref_path)
        
        # return {"loss": loss}

    def on_test_epoch_end(self):
        import yaml
        mean_SiSDRi = statistics.mean(self.SiSDRi)
        mean_SiSDR = statistics.mean(self.SiSDR)
        mean_SNR = statistics.mean(self.SNR)
        mean_SDR = statistics.mean(self.SDR)
        print(f'mean SiSDRi: {mean_SiSDRi}, mean SiSDR: {mean_SiSDR}, mean SNR: {mean_SNR}, mean SDR: {mean_SDR}')
        
        std_SiSDRi = statistics.stdev(self.SiSDRi)
        std_SiSDR = statistics.stdev(self.SiSDR)
        std_SNR = statistics.stdev(self.SNR)
        std_SDR = statistics.stdev(self.SDR)
        print(f'std SiSDRi: {std_SiSDRi}, std SiSDR: {std_SiSDR}, std SNR: {std_SNR}, std SDR: {std_SDR}')
        
        median_SiSDRi = statistics.median(self.SiSDRi)
        median_SiSDR = statistics.median(self.SiSDR)
        median_SNR = statistics.median(self.SNR)
        median_SDR = statistics.median(self.SDR)
        print(f'median SiSDRi: {median_SiSDRi}, median SiSDR: {median_SiSDR}, median SNR: {median_SNR}, median SDR: {median_SDR}')
        
        
        
        
        metrics_ins = {}
        for key, value in self.SiSDRi_ins.items():
            num_sample = len(value)
            if len(value) >= 2:
                std_SiSDRi_ins = statistics.stdev(value),
                std_SiSDR_ins = statistics.stdev(self.SiSDR_ins[key])
                std_SNR_ins = statistics.stdev(self.SNR_ins[key])
                std_SDR_ins = statistics.stdev(self.SDR_ins[key])
            else:
                std_SiSDRi_ins = 'n/a'
                std_SiSDR_ins = 'n/a'
                std_SNR_ins = 'n/a'
                std_SDR_ins = 'n/a'
            metrics_ins[key] = {
                'mean_SiSDRi': statistics.mean(value),
                'std_SiSDRi': std_SiSDRi_ins,
                'median_SiSDRi': statistics.median(value),
                'mean_SiSDR': statistics.mean(self.SiSDR_ins[key]),
                'std_SiSDR': std_SiSDR_ins,
                'median_SiSDR': statistics.median(self.SiSDR_ins[key]),
                'mean_SNR': statistics.mean(self.SNR_ins[key]),
                'std_SNR': std_SNR_ins,
                'median_SNR': statistics.median(self.SNR_ins[key]),
                'mean_SDR': statistics.mean(self.SDR_ins[key]),
                'std_SDR': std_SDR_ins,
                'median_SDR': statistics.median(self.SDR_ins[key]),
                'num_sample': num_sample
            }
        
        data = {
            'mean_SiSDRi': mean_SiSDRi,
            'std_SiSDRi': std_SiSDRi,
            'median_SiSDRi': median_SiSDRi,
            'mean_SiSDR': mean_SiSDR,
            'std_SiSDR': std_SiSDR,
            'median_SiSDR': median_SiSDR,
            'mean_SNR': mean_SNR,
            'std_SNR': std_SNR,
            'median_SNR': median_SNR,
            'metrics_ins': metrics_ins,
            'mean_SDR': mean_SDR,
            'std_SDR': std_SDR,
            'median_SDR': median_SDR
        }
        
        metrics_save_dir = os.path.join(self.logger.save_dir, 'metrics')
        os.makedirs(metrics_save_dir, exist_ok=True)
        metrics_save_path = os.path.join(metrics_save_dir, 'metrics.yaml')
        with open(metrics_save_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
                
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

