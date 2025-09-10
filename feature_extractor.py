"""
Created on Aug 26, 2025.
feature_extractor.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from transformers import AutoImageProcessor, AutoModel
import cv2
from tqdm import tqdm

from config.serde import open_experiment, create_experiment, delete_experiment, write_config

import warnings
warnings.filterwarnings('ignore')





def vindr_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)


    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'vindr-cxr1')
    org_df = pd.read_csv(os.path.join(file_base_dir, "master_list.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    subset_df = org_df[org_df['split'] == 'train']
    file_base_dir = os.path.join(file_base_dir, 'train')

    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        image_id = row['image_id']
        full_path = os.path.join(file_base_dir, image_id + '.jpg')
        img = cv2.imread(full_path)  # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            torch.save(features, full_path)



    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'vindr-cxr1')
    org_df = pd.read_csv(os.path.join(file_base_dir, "PATH_master_list.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    subset_df = org_df[org_df['split'] == 'test']
    file_base_dir = os.path.join(file_base_dir, 'test')

    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        image_id = row['image_id']
        full_path = os.path.join(file_base_dir, image_id + '.jpg')
        img = cv2.imread(full_path)  # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            torch.save(features, full_path)




def padchest_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)


    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'padchest')
    org_df = pd.read_csv(os.path.join(file_base_dir, "master.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        image_id = row['ImageID']
        subset = row['ImageDir']
        full_path = os.path.join(file_base_dir, str(subset), image_id)

        img = cv2.imread(full_path) # (h, w, d)

        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".png", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            torch.save(features, full_path)




def cxr14_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)


    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'NIH_ChestX-ray14')
    org_df = pd.read_csv(os.path.join(file_base_dir, "final_cxr14_master_list.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'CXR14', 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'CXR14', 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'CXR14', 'preprocessed1024')

    org_df = org_df[org_df['split'] == 'test']

    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        img_rel_path = row['img_rel_path']
        full_path = os.path.join(file_base_dir, img_rel_path)
        img = cv2.imread(full_path) # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".png", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            torch.save(features, full_path)



def UKA_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)


    file_base_dir = params['file_path']

    file_base_dir = os.path.join(file_base_dir, 'UKA_CXR')
    org_df = pd.read_csv(os.path.join(file_base_dir, "labels/master.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        image_id = row['image_id']
        subset = org_df[org_df['image_id'] == image_id]['subset'].values[0]
        full_path = os.path.join(file_base_dir, subset, str(image_id) + '.jpg')
        img = cv2.imread(full_path) # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            torch.save(features, full_path)



def chexpert_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)

    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    org_df = pd.read_csv(os.path.join(file_base_dir, "CheXpert-v1.0", "master.csv"), sep=',')

    org_df = org_df[org_df['view'] == 'Frontal']

    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        img_path = os.path.join(file_base_dir, row['jpg_rel_path'])

        if image_size == 224:
            full_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed224/")
        elif image_size == 512:
            full_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed/")
        elif image_size == 1024:
            full_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed1024/")

        img = cv2.imread(full_path) # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            torch.save(features, full_path)




def mimic_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)

    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, "MIMIC")
    org_df = pd.read_csv(os.path.join(file_base_dir, "master.csv"), sep=',')

    PAview = org_df[org_df['view'] == 'PA']
    APview = org_df[org_df['view'] == 'AP']
    org_df = pd.concat([PAview, APview], ignore_index=True)

    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        img_path = os.path.join(file_base_dir, row['jpg_rel_path'])

        if image_size == 224:
            full_path = img_path.replace("/files/", "/preprocessed224/")
        elif image_size == 512:
            full_path = img_path.replace("/files/", "/preprocessed/")
        elif image_size == 1024:
            full_path = img_path.replace("/files/", "/preprocessed1024/")

        img = cv2.imread(full_path) # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            torch.save(features, full_path)




def pedi_feature_extractor(global_config_path="/PATH/vit-med/config/config.yaml",
                 experiment_name='name', image_size=224):
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        attn_implementation="sdpa"
    )
    model = model.to(device)

    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'vindr-pcxr')
    org_df = pd.read_csv(os.path.join(file_base_dir, "master.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    org_df = org_df[org_df['split'] == 'train']
    file_base_dir = os.path.join(file_base_dir, 'train')
    for idx, row in tqdm(org_df.iterrows(), total=len(org_df)):
        image_id = row['image_id']
        full_path = os.path.join(file_base_dir, image_id + '.jpg')
        img = cv2.imread(full_path)  # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            torch.save(features, full_path)



    file_base_dir = params['file_path']
    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_512')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed_1024')

    file_base_dir = os.path.join(file_base_dir, 'vindr-pcxr')
    org_df = pd.read_csv(os.path.join(file_base_dir, "master.csv"), sep=',')

    if image_size == 224:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed224')
    elif image_size == 512:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed')
    elif image_size == 1024:
        file_base_dir = os.path.join(file_base_dir, 'preprocessed1024')

    subset_df = org_df[org_df['split'] == 'test']
    file_base_dir = os.path.join(file_base_dir, 'test')
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        image_id = row['image_id']
        full_path = os.path.join(file_base_dir, image_id + '.jpg')
        img = cv2.imread(full_path)  # (h, w, d)
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        image = trans(img)
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            features = output.last_hidden_state.mean(dim=1).squeeze(0)
            if image_size == 224:
                full_path = full_path.replace("/preprocessed224/", "/dinov3_feats_preprocessed224/")
            elif image_size == 512:
                full_path = full_path.replace("/preprocessed/", "/dinov3_feats_preprocessed512/")
            elif image_size == 1024:
                full_path = full_path.replace("/preprocessed1024/", "/dinov3_feats_preprocessed1024/")
            full_path = full_path.replace(".jpg", ".pt")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            torch.save(features, full_path)


