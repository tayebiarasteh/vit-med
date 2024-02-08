# Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images


Overview
------

* This is the official repository of the paper [**Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images**](https://doi.org/10.1186/s41747-023-00411-3).
* Update: We have released our fine-tuned network weights for research purposes. Check [this link](https://www.dropbox.com/scl/fi/6da3721irzs4swhhjlffi/networks.zip?rlkey=7r1wokvofq5gl5eaykyxvif0k&dl=0)! (Size: 1.6 GB)

Abstract
------
Pre-training datasets, like ImageNet, have become the gold standard in medical image analysis. However, the emergence of self-supervised learning (SSL), which leverages unlabeled data to learn robust features, presents an opportunity to bypass the intensive labeling process. In this study, we explored if SSL for pre-training on non-medical images can be applied to chest radiographs and how it compares to supervised pre-training on non-medical images and on medical images. We utilized a vision transformer and initialized its weights based on (i) SSL pre-training on natural images (DINOv2), (ii) SL pre-training on natural images (ImageNet dataset), and (iii) SL pre-training on chest radiographs from the MIMIC-CXR database. We tested our approach on over 800,000 chest radiographs from six large global datasets, diagnosing more than 20 different imaging findings. Our SSL pre-training on curated images not only outperformed ImageNet-based pre-training (P<0.001 for all datasets) but, in certain cases, also exceeded SL on the MIMIC-CXR dataset. Our findings suggest that selecting the right pre-training strategy, especially with SSL, can be pivotal for improving artificial intelligence (AI)'s diagnostic accuracy in medical imaging. By demonstrating the promise of SSL in chest radiograph analysis, we underline a transformative shift towards more efficient and accurate AI models in medical imaging.


### Prerequisites

The software is developed in **Python 3.9**. For the deep learning, the **PyTorch 2.0** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate enhancingpaper
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. Everything can be run from *./main_vitmed.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./Train_Valid_vitmed.py* contains the training and validation processes.
* *./Prediction_vitmed.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper:

S. Tayebi Arasteh, L. Misera, J.N. Kather  et al. *Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images*. Eur Radiol Exp 8, 10 (2024). https://doi.org/10.1186/s41747-023-00411-3.

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
