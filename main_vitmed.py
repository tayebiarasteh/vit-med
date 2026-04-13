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

import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login




def main_train_2D(global_config_path="/PATH/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', pretrained=False, vit=False, dino=True, image_size=224, batch_size=30, lr=1e-5):
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
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
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
        valid_dataset = pedicxr_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)

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
    if vit:
        if dino:

            #dinov3 and v2 models
            model = AutoModel.from_pretrained(
                # "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                # "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov2-base",
                # torch_dtype=torch.float16,
                # device_map="auto",
                attn_implementation="sdpa"
            )
            # model.head = torch.nn.Linear(in_features=4096, out_features=len(weight))
            # model.head = torch.nn.Linear(in_features=768, out_features=len(weight), dtype=torch.float16)
            model.head = torch.nn.Linear(in_features=768, out_features=len(weight))


            # convnext model
            # model = AutoModel.from_pretrained(
            #     "facebook/dinov3-convnext-base-pretrain-lvd1689m",     # dinov3
            #     # "facebook/convnext-base-224-22k", # imagenet
            #     use_safetensors=True
            # )
            # model.head = torch.nn.Linear(in_features=1024, out_features=len(weight))

        else:
            model = load_pretrained_timm_model(num_classes=len(weight), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d', pretrained=pretrained)

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    if vit:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                      weight_decay=float(params['Network']['weight_decay']))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr),
                                     weight_decay=float(params['Network']['weight_decay']),
                                     amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)

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
        valid_dataset = vindr_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)
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
        valid_dataset = pedicxr_feat_loader(cfg_path=cfg_path, mode='valid', augment=False, image_size=image_size)

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








