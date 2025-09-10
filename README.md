# Self-supervised and supervised pretraining for chest radiograph AI

## Papers in this repository

1) **Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images**  
European Radiology Experimental, 2024.  
DOI: https://doi.org/10.1186/s41747-023-00411-3

2) **High-resolution self-supervised learning with DINOv3 advances chest radiograph analysis**


## Prerequisites

The software is developed in **Python 3.9**. For deep learning, the **PyTorch 2.8** framework is used.

Main Python modules required for the software can be installed from `./requirements.yaml`:

```
$ conda env create -f requirements.yaml
$ conda activate vitmed
```

**Note:** This might take a few minutes.


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

S. Tayebi Arasteh, et al. High-resolution self-supervised learning with DINOv3 advances chest radiograph analysis. 2025.

```bibtex
@article{dinov3_cxr_2025,
  author  = {Tayebi Arasteh, Soroosh and others},
  title   = {High-resolution self-supervised learning with DINOv3 advances chest radiograph analysis},
  year    = {2025},
  doi     = {},
  url     = {}
}