<div align="center">
# GuideSep

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>


## Table Of Contents

- [Description](#description)
- [Setup](#setup)
    * [Install dependencies](#install-dependencies)
    * [Hydra-lightning](#install-dependencies)
- [How to run](#how-to-run)
    * [Run experiment and evaluation](#run-experiment-and-evaluation)
    * [Examples](#examples)
    * [Demo page](#demo-page)
- [Diffusion components](#diffusion-components)
    * [1.Diffusion](#1diffusion)

## Description

A pedagogical diffuser codebase designed for audio using denoising score matching formulation discussed in EDM. 
We have included a collection or (re)implementations of diffusion models, together with VAE, VQ-GAN for various audio applications. 
This repository uses hydra-lightning config management and is suitable for developping new models efficiently.
Since the diffusion process can be shared across different tasks, we will try to cover various audio based applications.
Additionally, this repo can be used as a learning resource for diffusion models which includes detailed docstrings linked to the paper, comments, and notebooks for introducing diffusion models both theoretically and practically.


## Setup

### Install dependencies

```bash
# clone project
git clone https://github.com/YutongWen/GuideSep
cd GuideSep

# [OPTIONAL] create conda environment
conda create -n guidesep python=3.8
conda activate guidesep

# install pytorch (>=2.0.1):
conda install pytorch torchvision torchaudio -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```
### Hydra-lightning

A config management tool that decouples dataloaders, training, network backbones etc.

## How to run

### Run experiment and evaluation
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash ddp mixed precision
CUDA_VISIBLE_DEVICES=0,3 python src/train.py trainer=ddp.yaml trainer.devices=2 experiment=example.yaml +trainer.precision=16-mixed +trainer.accumulate_grad_batches=4
```

For RTX 4090, add `NCCL_P2P_DISABLE=1` ([verified, ref here](https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366)) otherwise, DDP will stuck.

Or train model with  single GPU resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python src/train.py experiment=example1.yaml +trainer.precision=16-mixed ckpt_path="/path/to/ckpt/name.ckpt"
```

Or evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python src/eval.py ckpt_path='dummy.ckpt' +trainer.precision=16 experiment=example2.yaml
```

Particularly, grid search for tuning hyperparameters during sampling:

```bash
CUDA_VISIBLE_DEVICES=2 python src/eval.py --multirun ckpt_path='ckpt.pt' +trainer.precision=16-mixed experiment=experiment.yaml model.sampler.param1=3,6,9 model.sampler.param2=1.0,1.1
```

<!-- ### Examples
We list implemented "essential oils" for the audio diffuser, the following example recipes are trained and verified.

| **Model**   | **Dataset**|**Pytorch-lightning Script** |**Config** |
|------------|------------|--------------------------|-------------------|
|Diff-UNet-Waveform | SC09|[diffunet_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/diffunet_module.py) | [diffunet_sc09.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/diffunet_sc09.yaml)|
|Diff-UNet-Complex | SC09|[diffunet_complex_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/diffunet_complex_module.py) | [diffunet_complex_sc09.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/diffunet_complex_sc09.yaml)|
|Diff-UNet-Complex-VP | SC09|[diffunet_complex_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/diffunet_complex_module.py) | [diffunet_complex_sc09_vp.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/diffunet_complex_sc09_vp.yaml)|
|Diff-UNet-Complex-V-objective | SC09|[diffunet_complex_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/diffunet_complex_module.py) | [diffunet_complex_sc09_vobj.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/diffunet_complex_sc09_vobj.yaml)|
|Diff-UNet-Complex-CFG | DCASE2023-task7|[diffunet_complex_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/diffunet_complex_module.py) | [diffunet_complex_dcaseDev_cfg.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/diffunet_complex_dcaseDev_cfg.yaml)|
| VQ-GAN(WIP)|VCTK|[vqgan_module.py](https://github.com/gzhu06/AudioDiffuser/blob/main/src/models/vqgan_module.py) |[vqgan1d_vctk.yaml](https://github.com/gzhu06/AudioDiffuser/blob/main/configs/experiment/vqgan1d_vctk.yaml)| -->

### Demo Page
We generate samples (if any) from pretrained models in [example section](#examples), hosted in the branch [web_demo](https://github.com/gzhu06/AudioDiffuser/tree/web_demo) at [https://gzhu06.github.io/AudioDiffuser/](https://gzhu06.github.io/AudioDiffuser/).

## Diffusion components
In this repo, we mainly use the desnoising score matching formulation discussed in [EDM](https://github.com/NVlabs/edm) based on [Audio Diffusion release v0.0.94](https://github.com/archinetai/audio-diffusion-pytorch/releases/tag/v0.0.94). In EDM, a common framework is proposed for many diffusion models and decouples the design of sampling schedule, noise level parameterization etc.

### 1.Diffusion

The forward function runs one neural function evaluation (NFE). In EDM, the diffusion model training involves pre-conditioning and loss weighting, inference involves denoising which calls `forward`.

```python
class Diffusion(nn.Module):
    
    @abstractmethod
    def loss_weight(self):
        pass
    
    @abstractmethod
    def get_scale_weights(self):
        pass

    def denoise_fn(self):
        pass

    def forward(self):
        pass

```

### 2.Sampler
Different samplers take different parameters 

### 3.Scheduler

### 4.Backbones

### 5. Generation Evaulation
We compare different frameworks by testing on sc09 dataset using [unconditional audio generation benchmark repo](https://github.com/gzhu06/Unconditional-Audio-Generation-Benchmark). 

### 6.Other Applications
Speech enhancement
Super resolution
source separation
vocoder
super-resolution


## TODO
### Code
- [ ] use flash attention (torch version)
- [ ] add pooled embeddings
- [ ] post-hoc ema
- [ ] checkpointing
- [ ] add dpmv3
- [ ] add wavelet
- [ ] consistency models
- [ ] applications: source separation
- [ ] discrete diffusion

<!-- ### Notebooks TODO
- [ ] Sampler: ADM sampling
- [ ] Diffusion
- [ ] Scheduler

### Check TODO
- Diffusion: ADM sampling -->

## Notebooks

We listed our note on diffusion models.

## Code References
- [k-Diffusion by Katherine Crowson](https://github.com/crowsonkb/k-diffusion)
- [EDM by Nvdia](https://github.com/NVlabs/edm)
- [Audio Diffusion by Flavio](https://github.com/archinetai/audio-diffusion-pytorch)
- [EfficientUNet by lucidrians](https://github.com/lucidrains/imagen-pytorch)
- [UNet in ADM by openai](https://github.com/openai/guided-diffusion)
- [Unconditional diffwave by philsyn](https://github.com/philsyn/DiffWave-unconditional)


## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).