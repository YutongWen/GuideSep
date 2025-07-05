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
    * [FluidSynth](#install-dependencies)
- [Model Checkpoint](#model-checkpoint)
- [How to run](#how-to-run)
    * [Run experiment and evaluation](#run-experiment-and-evaluation)
    * [Demo page](#demo-page)

## Description

Music source separation (MSS) aims to extract individual instrument sources from their mixture. While most existing methods focus on the widely adopted four-stem separation setup (vocals, bass, drums, and other instru- ments), this approach lacks the flexibility needed for real-world applications. To address this, we propose GuideSep, a diffusion-based MSS model capable of instrument-agnostic separation beyond the four-stem setup. GuideSep is conditioned on multiple inputs: a waveform mimicry condition, which can be easily provided by humming or playing the target melody, and mel-spectrogram domain masks, which offer additional guidance for separation. Unlike prior approaches that relied on fixed class labels or sound queries, our conditioning scheme, coupled with the generative approach, provides greater flexibility and applicability. Additionally, we design a mask-prediction baseline using the same model architecture to systematically compare predictive and generative approaches. Our objective and subjective evaluations demonstrate that GuideSep achieves high-quality separation while enabling more versatile instrument extraction, highlighting the potential of user participation in the diffusion-based generative process for MSS.


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

Please change the root directory in the `configs/paths/default.yaml`.

### FluidSynth

Optional: install [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download) for training data simulation. You don't need this for inference with your own uploaded audio. 

Optional: for virtual instruments, we use [Aegean Symphonic Orchestra](https://sites.google.com/view/hed-sounds/aegean-symphonic-orchestra). Please download the Soundfont sf2 and put it in the project directory.

## Model Checkpoint

Model checkpoint is available at [Hugging Face](https://huggingface.co/YutongCooper/GuideSep-v1)

## How to run
### Run inference with user provided samples
A simple inference scipt is available at `inference.ipynb`. This notebook loads the provided model checkpoint, and does separation using a provided example from `mix/` and `cond/`. The script implements a simple UI for mask sketching. You can provide your own masks for separation. 

The diffusion inference scheduler and sampler configs are read from `configs/experiment/diffunet_complex_target_ins_extraction.yaml`. You can change the sampler config directly in the yaml file.

### Run experiment and evaluation
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash ddp mixed precision
CUDA_VISIBLE_DEVICES=0,3 python src/train.py trainer=ddp.yaml trainer.devices=2 experiment=diffunet_complex_target_ins_extraction.yaml +trainer.precision=16-mixed +trainer.accumulate_grad_batches=4
```

For RTX 4090, add `NCCL_P2P_DISABLE=1` ([verified, ref here](https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366)) otherwise, DDP will stuck.

Or train model with  single GPU resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python src/train.py experiment=diffunet_complex_target_ins_extraction.yaml +trainer.precision=16-mixed ckpt_path="/path/to/ckpt/name.ckpt"
```

Or evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python src/eval.py ckpt_path='dummy.ckpt' +trainer.precision=16 experiment=diffunet_complex_target_ins_extraction_eval.yaml
```

### Demo Page
We show demos of our model performing separation on real-world music with user-input humming as well as samples under evaluation setup. The dmeo page is hosted in the branch [demo](https://github.com/YutongWen/GuideSep/tree/demo) at [https://yutongwen.github.io/GuideSep/](https://yutongwen.github.io/GuideSep/).

## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).