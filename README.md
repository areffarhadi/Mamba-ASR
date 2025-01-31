# ConMamba [![arXiv](https://img.shields.io/badge/arXiv-2407.09732-<COLOR>.svg)](https://arxiv.org/abs/2407.09732)

This is a fork of an implementation of convolution-augmented Mamba for speech recognition.

## Changes in this Fork
we changed the structure of the data preparation code and prepared two training strategies: <be> 
1. training Mamba ASR for any dataset <be>
2. training MAmba ASR for a mix of Librispeech and any other datasets together (recommended)

You can modify the YAML file to change the architecture of the model.

## New training
To train a ConMamba Encoder-Mamba Decoder model on one GPU for any dataset:
```
python train_S2S_dataset.py hparams/S2S/conmambamamba_small_dataset.yaml --data_folder <YOUR_PATH_TO_LIBRISPEECH> --precision bf16 
```
To train a ConMamba Encoder-Mamba Decoder model on one GPU for mixing any dataset with Librispeech to improve the performance:
```
python train_S2S.py hparams/S2S/conmambamamba_small_mix.yaml --data_folder <YOUR_PATH_TO_LIBRISPEECH> --precision bf16 
```

## Architecture

<img src="figures/conmamba.png" alt="conmamba" width="80%">
<img src="figures/mamba_encoder_decoder.png" alt="layers" width="80%">

## Prerequisites

1. Download LibriSpeech [corpus](https://www.openslr.org/12).

2. Install Packages.
```
conda create --name Slytherin python=3.9
conda activate Slytherin
pip install -r requirements.txt
```
You may need to install lower or higher versions of torch, torchaudio, causal-conv1d and mamba-ssm based on your hardware and system. Make sure they are compatible. 


## Training
To train a ConMamba Encoder-Transformer Decoder model on one GPU:
```
python train_S2S.py hparams/S2S/conmamba_large(small).yaml --data_folder <YOUR_PATH_TO_LIBRISPEECH> --precision bf16 
```
To train a ConMamba Encoder-Mamba Decoder model on one GPU:
```
python train_S2S.py hparams/S2S/conmambamamba_large(small).yaml --data_folder <YOUR_PATH_TO_LIBRISPEECH> --precision bf16 
```
To train a ConMamba Encoder model with a character-level CTC loss on four GPUs:
```
torchrun --nproc-per-node 4 train_CTC.py hparams/CTC/conmamba_large.yaml --data_folder <YOUR_PATH_TO_LIBRISPEECH> --precision bf16 
```

## Inference and Checkpoints (Later)

## Performance (Word Error Rate%)
<img src="figures/performance.png" alt="performance" width="60%">

## Acknowledgement

We acknowledge the wonderful work of [Mamba](https://arxiv.org/abs/2312.00752) and [Vision Mamba](https://arxiv.org/abs/2401.09417). We borrowed their implementation of [Mamba](https://github.com/state-spaces/mamba) and [bidirectional Mamba](https://github.com/hustvl/Vim). The training recipes are adapted from [SpeechBrain](https://speechbrain.github.io).

## Citation
If you find this work helpful, please consider citing:

```bibtex
@misc{jiang2024speechslytherin,
      title={Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis}, 
      author={Xilin Jiang and Yinghao Aaron Li and Adrian Nicolas Florea and Cong Han and Nima Mesgarani},
      year={2024},
      eprint={2407.09732},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.09732}, 
}
```

You may also like our Mamba for speech separation: https://github.com/xi-j/Mamba-TasNet

