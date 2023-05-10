"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import matplotlib.pyplot as plt
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15




class vindr_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-cxr1')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_officialsoroosh_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_officialsoroosh_master_list.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']
            self.file_base_dir = os.path.join(self.file_base_dir, 'test')

        self.file_path_list = list(self.subset_df['image_id'])

        #### for comparisons #####
        # self.chosen_labels = ['No finding', 'Pneumonia'] # for comparison to VinDr-pcxr
        # self.chosen_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Lung Opacity', 'Pleural effusion', 'Pneumothorax', 'Pneumonia', 'No finding'] # for comparison to chexpert/mimic
        # self.chosen_labels = ['Atelectasis', 'Cardiomegaly', 'Pleural effusion', 'Infiltration', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Pulmonary fibrosis', 'Pleural thickening', 'No finding'] # for comparison to CXR14
        # self.chosen_labels = ['Cardiomegaly', 'Pleural effusion', 'Atelectasis'] # for comparison to UKA
        # self.chosen_labels = ['No finding', 'Cardiomegaly', 'Pleural effusion', 'Pneumonia', 'Atelectasis', 'Consolidation', 'Pleural thickening', 'COPD', 'Pulmonary fibrosis', 'Emphysema', 'Nodule/Mass', 'Infiltration'] # for comparison to padchest
        #### for comparisons #####

        self.chosen_labels = ['Cardiomegaly', 'Pleural effusion', 'Pneumonia', 'Atelectasis', 'No finding']




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx] + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class vindr_pediatric_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-pcxr')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list_vindr-pcxr.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']
            self.file_base_dir = os.path.join(self.file_base_dir, 'test')

        self.file_path_list = list(self.subset_df['image_id'])

        # self.chosen_labels = ['No finding', 'Pneumonia'] # Test on vindr/mimic/chexpert/cxr14

        self.chosen_labels = ['Pneumonia', 'Pneumonia'] # for Pneumonia



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx] + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class chexpert_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.size224 = size224
        self.file_base_dir = self.params['file_path']
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "nothree_master_list_20percenttest.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "5000_nothree_master_list_20percenttest.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.subset_df = self.subset_df[self.subset_df['view'] == 'Frontal']
        self.file_path_list = list(self.subset_df['jpg_rel_path'])

        #### for comparisons #####
        # self.chosen_labels = ['no_finding', 'pneumonia'] # for comparison to VinDr-pcxr
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices'] # for comparison to mimic
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'consolidation', 'lung_opacity', 'pleural_effusion', 'pneumothorax', 'pneumonia', 'no_finding'] # Test on VinDr
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'pleural_effusion', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'no_finding'] # Test on CXR14
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis'] # for comparison to UKA
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis', 'no_finding', 'pneumonia', 'consolidation'] # for comparison to padchest
        #### for comparisons #####

        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']





    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        if self.size224:
            img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed224/")
        else:
            img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class mimic_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.size224 = size224
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, "MIMIC")
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "nothree_master_list_20percenttest.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        self.subset_df = PAview.append(APview)
        self.file_path_list = list(self.subset_df['jpg_rel_path'])

        #### for comparisons #####
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'consolidation', 'lung_opacity', 'pleural_effusion', 'pneumothorax', 'pneumonia', 'no_finding'] # Test on VinDr
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'pleural_effusion', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'no_finding'] # Test on CXR14
        # self.chosen_labels = ['no_finding', 'pneumonia'] # Test on VinDr-pcxr
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices'] # Test on chexpert / for comparison to chexpert
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis'] # for comparison to UKA
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis', 'no_finding', 'pneumonia', 'consolidation'] # for comparison to padchest
        #### for comparisons #####

        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])

        if self.size224:
            img_path = img_path.replace("/files/", "/preprocessed224/")
        else:
            img_path = img_path.replace("/files/", "/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class UKA_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA/chest_radiograph')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "DP_project_also_original/original_novalid_UKA_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed')

        self.file_path_list = list(self.subset_df['image_id'])

        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'healthy'] # 5 labels




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]['subset'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, subset, str(self.file_path_list[idx]) + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]

        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            if self.chosen_labels[idx] == 'cardiomegaly':
                if int(label_df[self.chosen_labels[idx]].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 4:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 1:
                    label[idx] = 0
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 2:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'pleural_effusion':
                if int(label_df['pleural_effusion_right'].values[0]) == 3 or int(label_df['pleural_effusion_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['pleural_effusion_right'].values[0]) == 4 or int(label_df['pleural_effusion_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'atelectasis':
                if int(label_df['atelectasis_right'].values[0]) == 3 or int(label_df['atelectasis_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['atelectasis_right'].values[0]) == 4 or int(label_df['atelectasis_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'pneumonia':
                if int(label_df['pneumonic_infiltrates_right'].values[0]) == 3 or int(label_df['pneumonic_infiltrates_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['pneumonic_infiltrates_right'].values[0]) == 4 or int(label_df['pneumonic_infiltrates_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            if diseases == 'pleural_effusion':
                disease_length = sum(train_df['pleural_effusion_right'].values == 3)
                disease_length += sum(train_df['pleural_effusion_left'].values == 3)
                disease_length += sum(train_df['pleural_effusion_right'].values == 4)
                disease_length += sum(train_df['pleural_effusion_left'].values == 4)
            elif diseases == 'atelectasis':
                disease_length = sum(train_df['atelectasis_right'].values == 3)
                disease_length += sum(train_df['atelectasis_left'].values == 3)
                disease_length += sum(train_df['atelectasis_right'].values == 4)
                disease_length += sum(train_df['atelectasis_left'].values == 4)
            elif diseases == 'pneumonia':
                disease_length = sum(train_df['pneumonic_infiltrates_right'].values == 3)
                disease_length += sum(train_df['pneumonic_infiltrates_left'].values == 3)
                disease_length += sum(train_df['pneumonic_infiltrates_right'].values == 4)
                disease_length += sum(train_df['pneumonic_infiltrates_left'].values == 4)
            else:
                disease_length = sum(train_df[diseases].values == 3)
                disease_length += sum(train_df[diseases].values == 4)

            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class cxr14_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'NIH_ChestX-ray14')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_cxr14_master_list.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.file_path_list = list(self.subset_df['img_rel_path'])

        #### for comparisons #####
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'pneumonia', 'pneumothorax', 'consolidation', 'fibrosis', 'pleural_thickening', 'no_finding'] # for comparison to VinDr
        # self.chosen_labels = ['no_finding', 'pneumonia'] # for comparison to VinDr-pcxr
        # self.chosen_labels = ['atelectasis', 'cardiomegaly', 'effusion', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'no_finding'] # for comparison to chexpert/mimic
        # self.chosen_labels = ['cardiomegaly', 'effusion', 'atelectasis'] # for comparison to UKA
        # self.chosen_labels = ['cardiomegaly', 'effusion', 'atelectasis', 'infiltration', 'no_finding', 'pneumonia', 'fibrosis', 'emphysema', 'hernia', 'pleural_thickening', 'consolidation'] # for comparison to padchest
        #### for comparisons #####

        self.chosen_labels = ['cardiomegaly', 'effusion', 'pneumonia', 'atelectasis', 'no_finding']





    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx])) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['img_rel_path'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor


class padchest_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'padchest')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "padchest_master_list_20percenttest.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        APhorizview = self.subset_df[self.subset_df['view'] == 'AP_horizontal']
        self.subset_df = PAview.append(APview)
        self.subset_df = self.subset_df.append(APhorizview)
        self.file_path_list = list(self.subset_df['ImageID'])

        #### for comparisons #####
        # self.chosen_labels = ['no_finding', 'pneumonia'] # for comparison to VinDr-pcxr
        # self.chosen_labels = ['no_finding', 'cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'consolidation', 'pleural_thickening', 'COPD_signs', 'pulmonary_fibrosis', 'emphysema', 'nodule_mass', 'infiltrates'] # for comparison to VinDr-cxr
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis'] # for comparison to UKA
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis', 'infiltrates', 'no_finding', 'pneumonia', 'pulmonary_fibrosis', 'emphysema', 'hernia, 'pleural_thickening', 'consolidation'] # for comparison to cxr14
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'atelectasis', 'no_finding', 'pneumonia', 'consolidation'] # for comparison to mimic/chexpert
        #### for comparisons #####

        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']


    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]['ImageDir'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, str(subset), self.file_path_list[idx])) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        # casting to float16
        # img = img.half()
        # label = label.half()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor
