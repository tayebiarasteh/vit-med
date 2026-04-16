"""
Created on May 4, 2023.
main_vitmed.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import timm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from transformers import AutoImageProcessor, AutoModel

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_vitmed import Training
from Prediction_vitmed import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D, pedicxr_data_loader_2D
from data.feature_data_provider import vindr_feat_loader, padchest_feat_loader, cxr14_feat_loader, chexpert_feat_loader, pedicxr_feat_loader, mimic_feat_loader, UKA_feat_loader
from models.dinonet import DinoNet
from models.lora_wrappers import BackboneWithHead, _build_lora_model

import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login




def main_train_2D(global_config_path="/PATH/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', pretrained=False, model_name=False, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        train_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None


    # Changeable network parameters
    if model_name == 'vitb_dinov2':
        model = AutoModel.from_pretrained(
            "facebook/dinov2-base",
            attn_implementation="sdpa")
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))

    elif model_name == 'vitb_dinov3':
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            attn_implementation="sdpa")
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))

    elif model_name == 'vitb_imgnet':
        model = load_pretrained_timm_model(num_classes=len(weight), pretrained=pretrained, imgsize=image_size)

    elif model_name == 'convnext_dinov3':
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",  # dinov3
            use_safetensors=True)
        model.head = torch.nn.Linear(in_features=1024, out_features=len(weight))

    elif model_name == 'convnext_imgnet':
        model = AutoModel.from_pretrained(
            "facebook/convnext-base-224-22k", # imagenet
            use_safetensors=True)
        model.head = torch.nn.Linear(in_features=1024, out_features=len(weight))


    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr),
                                 weight_decay=float(params['Network']['weight_decay']),
                                 amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names, model_name=model_name)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])





def main_test_bootstrap(global_config_path="/PATH/config.yaml", experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='vindr', vit_imgnet=True, vit_dino2=True, vit_dino3=True, convnext_imgnet=True,
                        convnext_dino3=True, image_size=224, new_seed=False, bootstrap_index_path=None, n_bootstraps=1000):
    """Main function for multi label prediction with bootstrapping."""

    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        test_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    if vit_dino2:
        model = AutoModel.from_pretrained(
            "facebook/dinov2-base",
            attn_implementation="sdpa"
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))

    elif vit_dino3:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            attn_implementation="sdpa"
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))

    elif vit_imgnet:
        model = load_pretrained_timm_model(num_classes=len(weight), pretrained=True, imgsize=image_size)

    elif convnext_imgnet:
        model = AutoModel.from_pretrained(
            "facebook/convnext-base-224-22k",
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=1024, out_features=len(weight))

    elif convnext_dino3:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=1024, out_features=len(weight))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['Network']['batch_size'],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=16
    )

    stat_dir = os.path.join(params['target_dir'], params['stat_log_path'])
    os.makedirs(stat_dir, exist_ok=True)

    # Shared bootstrap indices across experiments for this dataset
    if bootstrap_index_path is None:
        bootstrap_dir = os.path.join(os.path.dirname(global_config_path), "bootstrap_indices")
        os.makedirs(bootstrap_dir, exist_ok=True)
        bootstrap_index_path = os.path.join(bootstrap_dir, f'bootstrapping_seeds_{dataset_name}.npy')

    if new_seed or not os.path.exists(bootstrap_index_path):
        index_list = np.random.choice(
            len(test_dataset),
            size=(n_bootstraps, len(test_dataset)),
            replace=True
        )
        np.save(bootstrap_index_path, index_list, allow_pickle=True)
    else:
        index_list = np.load(bootstrap_index_path, allow_pickle=True)

    if vit_dino2:
        vit_dino = True
    elif vit_dino3:
        vit_dino = True
    else:
        vit_dino = False

    if convnext_imgnet:
        convnext = True
    elif convnext_dino3:
        convnext = True
    else:
        convnext = False

    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)
    pred_array, target_array = predictor.predict_only(
        test_loader,
        vit_imgnet=vit_imgnet,
        vit_dino=vit_dino,
        convnext=convnext
    )

    pred_array = pred_array.cpu().numpy()
    target_array = target_array.int().cpu().numpy()

    np.save(os.path.join(stat_dir, f'pred_array_{dataset_name}.npy'), pred_array, allow_pickle=True)
    np.save(os.path.join(stat_dir, f'target_array_{dataset_name}.npy'), target_array, allow_pickle=True)

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])

    df.to_csv(
        os.path.join(stat_dir, f'predictions_teston_{dataset_name}.csv'),
        sep=',',
        index=False
    )

    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name)
    return AUC_list




def main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/PATH/config.yaml", experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
        dataset_name='vindr', image_size=224, bootstrap_index_path=None):
    """Calculate p-values from saved predictions using one shared bootstrap index file.
    """

    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        test_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)

    label_names = test_dataset.chosen_labels

    if bootstrap_index_path is None:
        bootstrap_dir = os.path.join(os.path.dirname(global_config_path), "bootstrap_indices")
        os.makedirs(bootstrap_dir, exist_ok=True)
        bootstrap_index_path = os.path.join(bootstrap_dir, f'bootstrapping_seeds_{dataset_name}.npy')

    if not os.path.exists(bootstrap_index_path):
        raise FileNotFoundError(
            f"Bootstrap index file not found at: {bootstrap_index_path}\n"
            f"Run main_test_bootstrap(..., new_seed=True) once first."
        )

    index_list = np.load(bootstrap_index_path, allow_pickle=True)

    predictor1 = Prediction(cfg_path1, label_names)
    stat_dir1 = os.path.join(params1['target_dir'], params1['stat_log_path'])

    pred_path1 = os.path.join(stat_dir1, f'pred_array_{dataset_name}.npy')
    target_path1 = os.path.join(stat_dir1, f'target_array_{dataset_name}.npy')

    if not os.path.exists(pred_path1) or not os.path.exists(target_path1):
        raise FileNotFoundError(
            f"Saved predictions for experiment '{experiment_name1}' not found.\n"
            f"Expected:\n{pred_path1}\n{target_path1}\n"
            f"Run main_test_bootstrap for this experiment first."
        )

    pred_array1 = np.load(pred_path1, allow_pickle=True)
    target_array1 = np.load(target_path1, allow_pickle=True)

    AUC_list1 = predictor1.bootstrapper(pred_array1, target_array1, index_list, dataset_name)

    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    stat_dir2 = os.path.join(params2['target_dir'], params2['stat_log_path'])

    pred_path2 = os.path.join(stat_dir2, f'pred_array_{dataset_name}.npy')
    target_path2 = os.path.join(stat_dir2, f'target_array_{dataset_name}.npy')

    if not os.path.exists(pred_path2) or not os.path.exists(target_path2):
        raise FileNotFoundError(
            f"Saved predictions for experiment '{experiment_name2}' not found.\n"
            f"Expected:\n{pred_path2}\n{target_path2}\n"
            f"Run main_test_bootstrap for this experiment first."
        )

    pred_array2 = np.load(pred_path2, allow_pickle=True)
    target_array2 = np.load(target_path2, allow_pickle=True)

    if target_array1.shape != target_array2.shape or not np.array_equal(target_array1, target_array2):
        raise ValueError(
            "The saved target arrays of the two experiments are not identical. "
            "Paired bootstrap comparison requires the exact same test set ordering."
        )

    AUC_list2 = predictor2.bootstrapper(pred_array2, target_array2, index_list, dataset_name)

    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')

    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)

    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    return AUC_list1, AUC_list2




def load_pretrained_timm_model(num_classes=2, model_name='vit_base_patch16_224_in21k', pretrained=False, imgsize=512):
    # Load a pre-trained model from config file
    if model_name == 'resnet50d':
    # if model_name == 'densenet121':
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    else:
        model = timm.create_model(model_name, num_classes=num_classes, img_size=imgsize, pretrained=pretrained)

    # model.load_state_dict(torch.load('/cluster/arasteh/Documents/Repositories_target_files/vit-med/mimicprets/for_cxrpadchest_res_imagnet_224/mimicpretraining_224.pth'))

    for param in model.parameters():
        param.requires_grad = True

    return model




def main_train_after_feature(global_config_path="/PATH/config.yaml", valid=False,
                  resume=False, experiment_name='name', dataset_name='vindr', image_size=224, batch_size=30, lr=1e-5):
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        train_dataset = vindr_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = vindr_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = chexpert_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = mimic_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = UKA_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = cxr14_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = padchest_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        train_dataset = pedicxr_feat_loader(cfg_path=cfg_path, mode='train', image_size=image_size)
        valid_dataset = pedicxr_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    model = DinoNet(out_features=len(weight))

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(params['Network']['weight_decay']))

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])




def main_test_head_bootstrap(global_config_path="/PATH/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='vindr', image_size=224):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        test_dataset = pedicxr_feat_loader(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    model = DinoNet(out_features=len(weight))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)
    pred_array, target_array = predictor.predict_only(test_loader, vit_imgnet=True, vit_dino=False, convnext=False)

    #########################################
    pred_array = pred_array.cpu().numpy()
    target_array = target_array.int().cpu().numpy()

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(params['target_dir'], params['stat_log_path']) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)

    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name)




def main_train_2D_lora(global_config_path="/PATH/config.yaml", valid=True, resume=False, augment=False,
                        experiment_name='name', dataset_name='mimic', model_name='vitb_dinov3_lora', image_size=512,
                        batch_size=16, lr=1e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Main function for LoRA-based training.
    """

    if resume is True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)

    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        train_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None


    model, trainer_model_name = _build_lora_model(model_name=model_name, num_classes=len(weight), lora_r=lora_r,
                                                  lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    model_info['peft_method'] = 'LoRA'
    model_info['peft_r'] = lora_r
    model_info['peft_alpha'] = lora_alpha
    model_info['peft_dropout'] = lora_dropout
    model_info['peft_model_name'] = model_name
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(lr), weight_decay=float(params['Network']['weight_decay']),
        amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names, model_name=trainer_model_name)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])






def make_vindr_noisy_csv(csv_path, out_path=None, noise_rate=0.10, seed=42, noise_mode="asymmetric"):
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)

    chosen_labels = [
        "Cardiomegaly",
        "Pleural effusion",
        "Pneumonia",
        "Atelectasis",
        "No finding",
        "Consolidation",
        "Pneumothorax",
        "Pleural thickening",
        "Lung Opacity",
        "Pulmonary fibrosis",
        "Nodule/Mass",
    ]

    required_cols = ["split"] + chosen_labels
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    noisy_df = df.copy()
    train_mask = noisy_df["split"].astype(str).str.lower() == "train"

    # We inject noise only into the 10 abnormal labels from the setup.
    noisy_target_labels = [c for c in chosen_labels if c != "No finding"]

    # These are columns whose non-null values are only 0/1.
    binary_cols = []
    for col in noisy_df.columns:
        vals = pd.Series(noisy_df[col]).dropna().unique().tolist()
        if len(vals) > 0 and all(v in (0, 1) for v in vals):
            binary_cols.append(col)

    if "No finding" not in binary_cols:
        raise ValueError("'No finding' was not detected as a binary label column.")

    # All pathology columns for recomputing No finding:
    # every binary label except 'No finding'
    all_abnormal_label_cols = [c for c in binary_cols if c != "No finding"]

    missing_binary = [c for c in noisy_target_labels if c not in all_abnormal_label_cols]
    if missing_binary:
        raise ValueError(
            f"These chosen abnormal labels were not detected as binary label columns: {missing_binary}"
        )

    summary = {
        "noise_rate": noise_rate,
        "noise_mode": noise_mode,
        "seed": seed,
        "train_rows": int(train_mask.sum()),
        "noisy_target_labels": noisy_target_labels,
        "all_abnormal_label_cols_for_no_finding": all_abnormal_label_cols,
        "label_flips_per_column": {},
    }

    # Store clean valid/test labels to ensure they remain unchanged
    untouched_mask = ~train_mask
    untouched_before = df.loc[untouched_mask, binary_cols].copy()

    for col in noisy_target_labels:
        values = noisy_df.loc[train_mask, col].to_numpy(copy=True)

        non_null = pd.Series(values).dropna().unique().tolist()
        bad = [x for x in non_null if x not in (0, 1)]
        if bad:
            raise ValueError(f"Column '{col}' contains non-binary values: {bad}")

        rand = rng.random(values.shape[0])

        if noise_mode == "symmetric":
            flip_mask = rand < noise_rate

        elif noise_mode == "asymmetric":
            # More realistic for weak labels:
            # positives are dropped more often than negatives become positives.
            pos_flip = (values == 1) & (rand < noise_rate)
            neg_flip = (values == 0) & (rand < (noise_rate / 4.0))
            flip_mask = pos_flip | neg_flip

        else:
            raise ValueError("noise_mode must be 'asymmetric' or 'symmetric'")

        flipped = values.copy()
        flipped[flip_mask] = 1 - flipped[flip_mask]
        noisy_df.loc[train_mask, col] = flipped

        summary["label_flips_per_column"][col] = {
            "n_train_examples": int(values.shape[0]),
            "n_flipped": int(flip_mask.sum()),
            "flip_fraction_over_train_rows": float(flip_mask.mean()),
            "n_positive_before": int((values == 1).sum()),
            "n_positive_after": int((flipped == 1).sum()),
        }

    train_abnormal_sum = noisy_df.loc[train_mask, all_abnormal_label_cols].sum(axis=1)
    noisy_df.loc[train_mask, "No finding"] = (train_abnormal_sum == 0).astype(int)

    untouched_after = noisy_df.loc[untouched_mask, binary_cols]
    if not untouched_before.equals(untouched_after):
        raise RuntimeError("Validation/test binary label columns were modified, which should never happen.")


    noisy_df.to_csv(out_path, index=False)
    summary["saved_to"] = out_path

    nf_before = df.loc[train_mask, "No finding"].sum()
    nf_after = noisy_df.loc[train_mask, "No finding"].sum()
    summary["no_finding_recomputed_on_train"] = {
        "n_positive_before": int(nf_before),
        "n_positive_after": int(nf_after),
    }

    return noisy_df, summary


