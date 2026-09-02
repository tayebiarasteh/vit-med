# Self-supervised and supervised pretraining for chest radiograph AI

## Papers in this repository

1) **Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images**  
European Radiology Experimental, 2024.  
DOI: https://doi.org/10.1186/s41747-023-00411-3

2) **Resolution-dependent self-supervised transfer in chest radiograph classification**  
Communications Medicine, 2026.  
DOI: https://doi.org/10.1038/s43856-026-01897-9



## Environment setup

Training and evaluation were performed strictly in FP32. 
Implementation details: Python 3.9 with PyTorch 2.8 and torchvision 0.23. Core libraries: NumPy 1.22, SciPy 1.10, scikit-learn 1.2, pandas 1.4, timm 0.6, and OpenCV (cv2) 4.7. Hugging Face tooling: transformers 4.56, huggingface-hub 0.34, datasets 2.19, accelerate 1.10, tokenizers 0.21, and safetensors 0.4.

### Prerequisites

The codebase was originally developed with earlier library versions; however, the configuration below provides a fully compatible and CUDA-enabled environment validated on modern NVIDIA GPUs.  
PyTorch is installed via official wheels with CUDA support to avoid dependency conflicts, and all remaining packages are installed through `pip`.  
No system-wide CUDA toolkit installation is required, as the PyTorch wheels bundle the necessary CUDA runtime.


```
$ conda create -n NAME python=3.11 -y
$ conda activate NAME
$ python -m pip install --upgrade pip
```

```
$ python -m pip install \
  torch==2.8 \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

```
$ python -m pip install \
  accelerate \
  transformers \
  tokenizers \
  safetensors \
  huggingface-hub \
  matplotlib \
  pandas \
  timm \
  tensorboardX \
  tqdm \
  jupyter \
  scikit-learn \
  opencv-python \
  opacus
```


---

## Model initializations used

**ImageNet (supervised):**
- ViT-B/16 (via timm): `vit_base_patch16_224_in21k`  
  https://github.com/huggingface/pytorch-image-models

**DINOv2 (self-supervised):**
- ViT-B/16: https://huggingface.co/facebook/dinov2-base

**DINOv3 (self-supervised):**
- ViT-B/16: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m  
- ConvNeXt-B: https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m  
- ViT-7B/16 (frozen features): https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m  
  *(ConvNeXt DINOv3 weights were loaded from SafeTensors.)*

---

## Code structure

- `main_vitmed.py` — single entry point for training/evaluation.  
- `configs/config.yaml` — edit data paths, preprocessing, model/backbone, initialization (ImageNet / DINOv2 / DINOv3), resolution (224 / 512), optimizer and schedule.  
- `data/` — dataset I/O, preprocessing, augmentation.  
- `Train_Valid_vitmed.py` — training / validation loops.  
- `Prediction_vitmed.py` — inference & metrics.

---

## Quickstart

1) Prepare datasets following the paths and splits in `configs/config.yaml`.  
2) Choose an `experiment` name; the script will create a folder with checkpoints, metrics, TensorBoard logs, and a copy of the effective config.  
3) Launch training/evaluation from the project root, e.g.

```
python main_vitmed.py --config ./configs/config.yaml --experiment dinov3_convnext_512
```

## In case you use this repository, please cite the original paper:

If you use this code, please cite **both** papers:

**Paper 1**  

S. Tayebi Arasteh, L. Misera, J.N. Kather, D. Truhn, S. Nebelung. *Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images*. European Radiology Experimental 8, 10 (2024). https://doi.org/10.1186/s41747-023-00411-3

### BibTex

    @article {enhancingarasteh,
      author = {Tayebi Arasteh, Soroosh and Misera, Leo and Kather, Jakob Nikolas and Truhn, Daniel and Nebelung, Sven},
      title = {Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images},
      year = {2024},
      volume = {8},
      number = {10},
      doi = {10.1186/s41747-023-00411-3},
      publisher = {Springer},
      URL = {https://doi.org/10.1186/s41747-023-00411-3},
      journal = {European Radiology Experimental}
    }

**Paper 2**

S. Tayebi Arasteh, et al. *Resolution-dependent self-supervised transfer in chest radiograph classification*. Communications Medicine, 6, 2026. https://doi.org/10.1038/s43856-026-01897-9.

```bibtex
@article{dinov3_cxr_2026,
  author  = {Soroosh Tayebi Arasteh and Mina Shaigan and Christiane Kuhl and Jakob Nikolas Kather and Sven Nebelung and Daniel Truhn},
  title   = {Resolution-dependent self-supervised transfer in chest radiograph classification},
  year    = {2026},
  volume = {6},
  journal = {Communications Medicine},
  doi     = {https://doi.org/10.1038/s43856-026-01897-9},
}
