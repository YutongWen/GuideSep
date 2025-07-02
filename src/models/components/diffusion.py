""" Diffusion Classes """

from math import pi
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from torch import Tensor
from .utils import extend_dim, clip, to_batch
from abc import abstractmethod

EPSI = 1e-7

class Diffusion(nn.Module):

    def __init__(
        self,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
    
    @abstractmethod
    def loss_weight(self):
        pass
    
    @abstractmethod
    def get_scale_weights(self):
        pass
    
    def denoise_fn(
        self,
        x_noisy: Tensor,
        net: nn.Module = None,
        inference: bool = False,
        cond_scale: float = 1.0,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs) -> Tensor:

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        
        # Predict network output
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy.ndim)

        # cfg interpolation during inference, skip during training
        if inference and cond_scale != 1.0:
            x_pred = net(c_in*x_noisy, c_noise, cond_drop_prob=0., **kwargs)
            
            null_logits = net(c_in*x_noisy, c_noise, cond_drop_prob=1., **kwargs)
            x_pred = null_logits + (x_pred - null_logits) * cond_scale
            
            # neg_logits = net(c_in*x_noisy, c_noise, torch.ones_like(x_classes)*6, 
            #                  cond_drop_prob=0., x_mask=x_mask, **kwargs)
            # x_pred = null_logits + (x_pred - neg_logits) * cond_scale
        
        else:
            x_pred = net(c_in*x_noisy, c_noise, **kwargs)

        # eq.7
        x_denoised = c_skip * x_noisy + c_out * x_pred
        
        # Clips in [-1,1] range, with dynamic thresholding if provided
        return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)
    
    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        device = x.device
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        # Compute denoised values
        x_noisy = x + sigmas_padded * noise
        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=device) * kwargs['x_mask'] + torch.ones(x.size(), device=device) * (~kwargs['x_mask'])*0.1
        else: 
            loss_mask = torch.ones(x.size(), device=device)

        x_denoised = self.denoise_fn(x_noisy=x_noisy, 
                                     net=net, sigmas=sigmas, 
                                     inference=inference, 
                                     cond_scale=cond_scale,
                                     **kwargs)
        
        # noise level weighted loss (weighted eq.2)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses * loss_mask, "b ... -> b", "sum")
        x_dim = list(range(len(x.shape)))
        losses = losses * self.loss_weight(sigmas) / torch.sum(torch.ones(x.size(), device=device), dim=tuple(x_dim[1:]))
        
        if 'mix_ref' in kwargs:
            reg = torch.sum(F.mse_loss(x_denoised[:, :2, :, :] + x_denoised[:, 2:, :, :], kwargs['mix_ref'], reduction="none"), dim=(1, 2, 3))
            reg = torch.where(kwargs['reg_mask'], torch.tensor(0.0), reg)
            reg = reduce(reg, "b ... -> b", "sum")
            reg = reg * kwargs['weight'] / torch.sum(torch.ones(x.size(), device=device), dim=tuple(x_dim[1:]))
            losses = losses + reg
        
        return losses # loss shape [B,]
    
class VEDiffusion(Diffusion):
    def __init__(
        self,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
        
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1
        c_noise = (0.5 * sigmas).log()
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = sigmas
        c_in = 1
        
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return 1 / (sigmas**2)
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)


class VPDiffusion(Diffusion):
    """VP Diffusion Models formulated by EDM"""

    def __init__(
        self,
        beta_min: float, 
        beta_d: float,
        M: float,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.M = M
        self.dynamic_threshold = dynamic_threshold

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return 1 / sigmas ** 2
    
    def t_to_sigma(self, t):
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def sigma_to_t(self, sigmas):
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigmas ** 2).log()).sqrt() - self.beta_min) / self.beta_d
    
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        c_noise = (self.M - 1) * self.sigma_to_t(sigmas)
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = - sigmas
        c_in = 1 / (sigmas ** 2 + 1).sqrt()
        return c_skip, c_out, c_in, c_noise
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)

    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        # ### TO FIX!!!
        # # Sample amount of noise to add for each batch element
        sigmas = self.t_to_sigma(sigmas)
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        # Compute denoised values
        x_noisy = x + sigmas_padded * noise
        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=x.device) * kwargs['x_mask'] + torch.ones(x.size(), device=x.device) * (~kwargs['x_mask'])*0.1
        else: 
            loss_mask = torch.ones(x.size(), device=x.device)
        x_denoised = self.denoise_fn(x_noisy, net, 
                                     sigmas=sigmas,
                                     inference=inference, 
                                     cond_scale=cond_scale,
                                     **kwargs)

        # noise level weighted loss (weighted eq.2)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses * loss_mask, "b ... -> b", "sum")
        losses = losses * self.loss_weight(sigmas) / torch.sum(torch.ones(x.size(), device=x.device), dim=(1,2,3))

        return losses

class EluDiffusion(Diffusion):
    """Elucidated Diffusion Models(EDM): https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        sigma_data: float,  # data distribution
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)
    
class VEluDiffusion(Diffusion):
    
    """ 
    in progress
    v-diffusion using EluDiffusion framework: 
    
    https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py#L9
    
    
    """

    def __init__(
        self,
        sigma_data: float = 1.0,  # data distribution
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:
        
        c_noise = self.sigma_to_t(sigmas)
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = -sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + self.sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return (t * pi / 2).tan()
    
    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size, device = x.shape[0], x.device
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)
        
        # Compute denoised values
        x_noisy = x + sigmas_padded * noise
        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=device) * kwargs['x_mask'] + torch.ones(x.size(), device=device) * (~kwargs['x_mask'])*0.1
        else: 
            loss_mask = torch.ones(x.size(), device=device)
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy.ndim)
        x_pred = net(c_in * x_noisy, c_noise, **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + EPSI)

        # Compute loss (need mask fixing)
        loss = F.mse_loss(x_pred, v_target) / torch.sum(loss_mask, dim=(1,2,3))
        return loss
    
class ReFlow(nn.Module):
    # Rectified flow training
    # Reference:
    #   https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
    # Noise: random noise

    def __init__(
        self,
        ln: bool = True,  # data distribution
        stratified: bool = True,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ):
        super().__init__()
        self.ln = ln
        self.stratified = stratified
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def transport_fn(self,
                     x_noisy: Tensor,
                     net: nn.Module = None,
                     inference: bool = False,
                     cond_scale: float = 1.0,
                     sigmas: Optional[Tensor] = None,
                     sigma: Optional[float] = None,
                     **kwargs) -> Tensor:

        batch_size, device = x_noisy.shape[0], x_noisy.device

        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # cfg interpolation during inference, skip during training
        if inference and cond_scale != 1.0:
            x_pred = net(x_noisy, sigmas, cond_drop_prob=0., **kwargs)
            
            null_logits = net(x_noisy, sigmas, cond_drop_prob=1., **kwargs)
            x_pred = null_logits + (x_pred - null_logits) * cond_scale
        
        else:
            x_pred = net(x_noisy, sigmas, **kwargs)
        
        return x_pred

    def forward(self, x: Tensor, 
                net: nn.Module, 
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size = x.size(0)
        if self.ln:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform

                quantiles = torch.linspace(0, 1, batch_size + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((batch_size,)).to(x.device) / batch_size
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                z = z * self.logit_std + self.logit_mean
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((batch_size,)).to(x.device) * self.logit_std + self.logit_mean
                t = torch.sigmoid(nt)
        else:
            # uniform [0, 1]
            t = torch.rand((batch_size,)).to(x.device)

        t_padded = extend_dim(t, dim=x.ndim)
        z1 = torch.randn_like(x)
        zt = (1 - t_padded) * x + t_padded * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)

        vtheta = self.transport_fn(zt, net, sigmas=t,
                                   inference=inference, 
                                   cond_scale=cond_scale,
                                   **kwargs)
        
        loss = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return loss, t
