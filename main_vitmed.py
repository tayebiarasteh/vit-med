"""
Created on May 4, 2023.
main_vitmed.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
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
from sklearn import metrics

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_vitmed import Training
from Prediction_vitmed import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D, cxr14_data_loader_2D, vindr_pediatric_data_loader_2D, padchest_data_loader_2D

import warnings
warnings.filterwarnings('ignore')




def main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', pretrained=False, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"

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

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'vindr_pediatric':
        train_dataset = vindr_pediatric_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_pediatric_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

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
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight))
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



def main_train_federated(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', train_sites=['vindr', 'cxr14'], pretrained=True, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """

        Parameters
        ----------
        global_config_path: str
            always global_config_path="FLspeech/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    train_loader = []
    valid_loader = []
    weight_loader = []
    loss_function_loader = []
    label_names_loader = []

    for dataset_name in train_sites:

        if dataset_name == 'vindr':
            train_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'vindr_pediatric':
            train_dataset_model = vindr_pediatric_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = vindr_pediatric_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'mimic':
            train_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'UKA':
            train_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'padchest':
            train_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

        model_info = params['Network']
        model_info['lr'] = lr
        model_info['batch_size'] = batch_size
        params['Network'] = model_info
        write_config(params, cfg_path, sort_keys=True)

        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels

        loss_function_model = BCEWithLogitsLoss

        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=batch_size,
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)

        train_loader.append(train_loader_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        # if dataset == 'cxr14':
        #     continue
        label_names_loader.append(label_names_model)

        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
        valid_loader.append(valid_loader_model)

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight_model))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight_model), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='resnet50d', pretrained=pretrained)

    trainer = Training(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        pass
        # trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_models(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, vit=vit)




def main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml", experiment_name='central_exp_for_test', epoch_num=5,
                 dataset_name='vindr', vit=False, dinov2=True, image_size=224):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'vindr_pediatric':
        test_dataset = vindr_pediatric_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
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
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight), imgsize=image_size)
    else:
        model = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=epoch_num)
    average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')
    print(f'\t model tested on the {dataset_name} test set\n')

    print(f'\t avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | avg specificity: {average_specificity.mean() * 100:.2f}%'
    f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n')

    print('Individual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('\nIndividual accuracy:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_accuracy[idx] * 100:.2f}%')

    print('\nIndividual sensitivity:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_sensitivity[idx] * 100:.2f}%')

    print('\nIndividual specificity:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_specificity[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'\t model tested on the {dataset_name} test set\n\n' \
          f'avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | avg specificity: {average_specificity.mean() * 100:.2f}%' \
          f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)

    msg = f'Individual AUROC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_AUROC[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual accuracy:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_accuracy[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual sensitivity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_sensitivity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual specificity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_specificity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)



def main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 experiment1_epoch_num=100, experiment2_epoch_num=100, dataset_name='vindr', vit=False, size224=False):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'vindr_pediatric':
        test_dataset = vindr_pediatric_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, size224=size224)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    if size224:
        imgsize = 224
    else:
        imgsize = 512

    # Changeable network parameters
    if vit:
        model1 = load_pretrained_timm_model(num_classes=len(weight), imgsize=imgsize)
    else:
        model1 = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model(model=model1, epoch_num=experiment1_epoch_num)
    pred_array1, target_array1 = predictor1.predict_only(test_loader)
    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list, dataset_name)

    # Changeable network parameters
    if vit:
        model2 = load_pretrained_timm_model(num_classes=len(weight), imgsize=imgsize)
    else:
        model2 = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)

    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    predictor2.setup_model(model=model2, epoch_num=experiment2_epoch_num)
    pred_array2, target_array2 = predictor2.predict_only(test_loader)
    AUC_list2 = predictor2.bootstrapper(pred_array2.cpu().numpy(), target_array2.int().cpu().numpy(), index_list, dataset_name)

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


def load_pretrained_timm_model(num_classes=2, model_name='vit_base_patch16_224', pretrained=False, imgsize=512):
    # Load a pre-trained model from config file

    if model_name == 'resnet50d':
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    else:
        model = timm.create_model(model_name, num_classes=num_classes, img_size=imgsize, pretrained=pretrained)
        # model.load_state_dict(
        #     torch.load('/home/arasteh/Documents/Repositories/vit-med/mimicpretraining_vitb16_imagenet_224.pth'))

    for param in model.parameters():
        param.requires_grad = True

    return model


def load_pretrained_dinov2(num_classes=2):
    # Load a pre-trained model from config file

    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.head = torch.nn.Linear(in_features=768, out_features=num_classes)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    for param in model.parameters():
        param.requires_grad = True

    return model






if __name__ == '__main__':
    delete_experiment(experiment_name='name', global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml")
    # main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml",
    #               valid=True, resume=False, augment=True, experiment_name='temp', dataset_name='vindr', pretrained=True, vit=True, size224=True, batch_size=32, lr=1e-5)
    # main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml",
    #               valid=True, resume=False, augment=True, experiment_name='padchest_resnet50_224_7labels_lr6e5', dataset_name='vindr',
    #                       pretrained=True, vit=False, dinov2=False, image_size=224, batch_size=128, lr=6e-5)

    main_train_federated(global_config_path="/home/soroosh/Documents/Repositories/vit-med/config/config.yaml",
                         valid=True, resume=False, augment=True, experiment_name='name', train_sites=['vindr', 'vindr'],
                         pretrained=True, vit=False, dinov2=False, image_size=224, batch_size=32, lr=1e-5)