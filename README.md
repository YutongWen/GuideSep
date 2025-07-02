# GuideSep

This is the official code implementation for the paper USER-GUIDED GENERATIVE SOURCE SEPARATION.

<div>

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
    * [FluidSynth](#install-dependencies)
- [How to run](#how-to-run)
    * [Run experiment and evaluation](#run-experiment-and-evaluation)
    * [Examples](#examples)
    * [Demo page](#demo-page)

## Description

Music source separation (MSS) aims to extract individual instrument sources from their mixture. While most existing methods focus on the widely adopted four-stem separation setup (vocals, bass, drums, and other instru- ments), this approach lacks the flexibility needed for real-world applications. To address this, we propose GuideSep, a diffusion-based MSS model capable of instrument-agnostic separation beyond the four-stem setup. GuideSep is conditioned on multiple inputs: a waveform mimicry condition, which can be easily provided by humming or playing the target melody, and mel-spectrogram domain masks, which offer additional guidance for separation. Unlike prior approaches that relied on fixed class labels or sound queries, our conditioning scheme, coupled with the generative approach, provides greater flexibility and applicability. Additionally, we design a mask-prediction baseline using the same model architecture to systematically compare predictive and generative approaches. Our objective and subjective evaluations demonstrate that GuideSep achieves high-quality separation while enabling more versatile instrument extraction, highlighting the potential of user participation in the diffusion-based generative pro- cess for MSS.


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

### FluidSynth

Install [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download) for training data simulation. You don't need this for inference with your own uploaded audio. 

For virtual instruments, we use [Aegean Symphonic Orchestra](https://sites.google.com/view/hed-sounds/aegean-symphonic-orchestra). Please download the Soundfont sf2 and put it in the project directory.

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

## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).