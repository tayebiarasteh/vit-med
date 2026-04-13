"""
Created on May 4, 2023.
Prediction_vitmed.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os.path
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from config.serde import read_config

epsilon = 1e-15




class Prediction:
    def __init__(self, cfg_path, label_names):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()



    def setup_cuda(self, cuda_device_id=0):
        """setup the device.
        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')





    def setup_model(self, model, model_file_name=None, epoch_num=100):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch" + str(epoch_num) + "_" + model_file_name))




    def predict_only(self, test_loader, vit_imgnet=True, vit_dino=True, convnext=True):
        """Evaluation with metrics epoch
        """
        self.model.eval()

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(tqdm(test_loader)):

            image = image.to(self.device)
            label = label.to(self.device)
            label = label.float()

            with torch.no_grad():
                if vit_imgnet:
                    output = self.model(image)  # for ViT imagenet

                elif vit_dino:
                    output = self.model(image)
                    output = self.model.head(output.last_hidden_state.mean(dim=1))  # for ViT dinov2 and v3

                elif convnext:
                    output = self.model(image)
                    output = self.model.head(output.pooler_output)  # for convnext (both dino & imagnet)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                labels_cache = torch.cat((labels_cache, label))

        return preds_with_sigmoid_cache, labels_cache




    def ci95(self, x, axis=0):
        """Return 2.5 and 97.5 percentiles along axis."""
        lo, hi = np.percentile(x, [2.5, 97.5], axis=axis)
        return lo, hi




    def bootstrapper(self, preds_with_sigmoid, targets, index_list, testsetname):
        def safe_auroc(y_true, y_score):
            if len(np.unique(y_true)) < 2:
                return np.nan
            return metrics.roc_auc_score(y_true, y_score)

        def safe_average_precision(y_true, y_score):
            if len(np.unique(y_true)) < 2:
                return np.nan
            return metrics.average_precision_score(y_true, y_score)

        def safe_pr_auc(y_true, y_score):
            if len(np.unique(y_true)) < 2:
                return np.nan
            precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_score)
            if recall_curve[0] > recall_curve[-1]:
                recall_curve = recall_curve[::-1]
                precision_curve = precision_curve[::-1]
            return metrics.auc(recall_curve, precision_curve)

        def nan_ci95(x, axis=0):
            lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=axis)
            return lo, hi

        def fmt_metric(mean, std, lo, hi):
            if np.isnan(mean):
                return "nan ± nan [95% CI: nan, nan]"
            return f"{mean * 100:.2f} ± {std * 100:.2f} [95% CI: {lo * 100:.2f}, {hi * 100:.2f}]"

        AUC_list = []
        AP_list = []
        PRAUC_list = []
        accuracy_list = []
        balanced_accuracy_list = []
        specificity_list = []
        sensitivity_list = []
        precision_list = []
        F1_list = []

        print('bootstrapping ... \n')

        n_bootstraps = len(index_list)

        for counter in range(n_bootstraps):
            bootstrap_indices = np.asarray(index_list[counter], dtype=np.int64)

            final_targets = targets[bootstrap_indices]
            final_preds_with_sigmoid = preds_with_sigmoid[bootstrap_indices]

            ############ Evaluation metric calculation ########

            # threshold finding for metrics calculation
            optimal_threshold = np.zeros(final_targets.shape[1])

            for idx in range(final_targets.shape[1]):
                if len(np.unique(final_targets[:, idx])) < 2:
                    optimal_threshold[idx] = 0.5
                else:
                    fpr, tpr, thresholds = metrics.roc_curve(
                        final_targets[:, idx],
                        final_preds_with_sigmoid[:, idx],
                        pos_label=1
                    )
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold[idx] = thresholds[optimal_idx]

            predicted_labels = (final_preds_with_sigmoid > optimal_threshold).astype(np.int32)

            # Metrics calculation (macro) over the whole set
            confusion = metrics.multilabel_confusion_matrix(final_targets, predicted_labels)

            F1_disease = []
            accuracy_disease = []
            balanced_accuracy_disease = []
            specificity_disease = []
            sensitivity_disease = []
            precision_disease = []
            AUROC_disease = []
            AP_disease = []
            PRAUC_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]

                sensitivity = TP / (TP + FN + epsilon)
                specificity = TN / (TN + FP + epsilon)
                precision = TP / (TP + FP + epsilon)
                accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
                f1 = 2 * TP / (2 * TP + FN + FP + epsilon)
                balanced_accuracy = (sensitivity + specificity) / 2.0

                F1_disease.append(f1)
                accuracy_disease.append(accuracy)
                balanced_accuracy_disease.append(balanced_accuracy)
                specificity_disease.append(specificity)
                sensitivity_disease.append(sensitivity)
                precision_disease.append(precision)

                AUROC_disease.append(safe_auroc(final_targets[:, idx], final_preds_with_sigmoid[:, idx]))
                AP_disease.append(safe_average_precision(final_targets[:, idx], final_preds_with_sigmoid[:, idx]))
                PRAUC_disease.append(safe_pr_auc(final_targets[:, idx], final_preds_with_sigmoid[:, idx]))

            average_f1_score = np.stack(F1_disease)
            average_AUROC = np.stack(AUROC_disease)
            average_AP = np.stack(AP_disease)
            average_PRAUC = np.stack(PRAUC_disease)
            average_accuracy = np.stack(accuracy_disease)
            average_balanced_accuracy = np.stack(balanced_accuracy_disease)
            average_specificity = np.stack(specificity_disease)
            average_sensitivity = np.stack(sensitivity_disease)
            average_precision = np.stack(precision_disease)

            AUC_list.append(average_AUROC)
            AP_list.append(average_AP)
            PRAUC_list.append(average_PRAUC)
            accuracy_list.append(average_accuracy)
            balanced_accuracy_list.append(average_balanced_accuracy)
            specificity_list.append(average_specificity)
            sensitivity_list.append(average_sensitivity)
            precision_list.append(average_precision)
            F1_list.append(average_f1_score)

        AUC_list = np.stack(AUC_list)
        AP_list = np.stack(AP_list)
        PRAUC_list = np.stack(PRAUC_list)
        accuracy_list = np.stack(accuracy_list)
        balanced_accuracy_list = np.stack(balanced_accuracy_list)
        specificity_list = np.stack(specificity_list)
        sensitivity_list = np.stack(sensitivity_list)
        precision_list = np.stack(precision_list)
        F1_list = np.stack(F1_list)

        # Per-class (individual) CIs
        auc_lo, auc_hi = nan_ci95(AUC_list, axis=0)
        ap_lo, ap_hi = nan_ci95(AP_list, axis=0)
        prauc_lo, prauc_hi = nan_ci95(PRAUC_list, axis=0)
        acc_lo, acc_hi = nan_ci95(accuracy_list, axis=0)
        balacc_lo, balacc_hi = nan_ci95(balanced_accuracy_list, axis=0)
        spec_lo, spec_hi = nan_ci95(specificity_list, axis=0)
        sens_lo, sens_hi = nan_ci95(sensitivity_list, axis=0)
        prec_lo, prec_hi = nan_ci95(precision_list, axis=0)
        f1_lo, f1_hi = nan_ci95(F1_list, axis=0)

        # Macro averages per bootstrap
        AUC_avg_per_bs = np.nanmean(AUC_list, axis=1)
        AP_avg_per_bs = np.nanmean(AP_list, axis=1)  # mAP
        PRAUC_avg_per_bs = np.nanmean(PRAUC_list, axis=1)
        ACC_avg_per_bs = np.nanmean(accuracy_list, axis=1)
        BALACC_avg_per_bs = np.nanmean(balanced_accuracy_list, axis=1)
        SPEC_avg_per_bs = np.nanmean(specificity_list, axis=1)
        SENS_avg_per_bs = np.nanmean(sensitivity_list, axis=1)
        PREC_avg_per_bs = np.nanmean(precision_list, axis=1)
        F1_avg_per_bs = np.nanmean(F1_list, axis=1)

        AUC_avg_lo, AUC_avg_hi = nan_ci95(AUC_avg_per_bs, axis=0)
        AP_avg_lo, AP_avg_hi = nan_ci95(AP_avg_per_bs, axis=0)
        PRAUC_avg_lo, PRAUC_avg_hi = nan_ci95(PRAUC_avg_per_bs, axis=0)
        ACC_avg_lo, ACC_avg_hi = nan_ci95(ACC_avg_per_bs, axis=0)
        BALACC_avg_lo, BALACC_avg_hi = nan_ci95(BALACC_avg_per_bs, axis=0)
        SPEC_avg_lo, SPEC_avg_hi = nan_ci95(SPEC_avg_per_bs, axis=0)
        SENS_avg_lo, SENS_avg_hi = nan_ci95(SENS_avg_per_bs, axis=0)
        PREC_avg_lo, PREC_avg_hi = nan_ci95(PREC_avg_per_bs, axis=0)
        F1_avg_lo, F1_avg_hi = nan_ci95(F1_avg_per_bs, axis=0)

        print('------------------------------------------------------'
              '----------------------------------')
        print('\t experiment:' + self.params['experiment_name'] + '\n')

        print(
            f"\t avg AUROC: {fmt_metric(np.nanmean(AUC_avg_per_bs), np.nanstd(AUC_avg_per_bs), AUC_avg_lo, AUC_avg_hi)} | "
            f"avg AP (mAP): {fmt_metric(np.nanmean(AP_avg_per_bs), np.nanstd(AP_avg_per_bs), AP_avg_lo, AP_avg_hi)} | "
            f"avg PR-AUC: {fmt_metric(np.nanmean(PRAUC_avg_per_bs), np.nanstd(PRAUC_avg_per_bs), PRAUC_avg_lo, PRAUC_avg_hi)}"
        )
        print(
            f"\t avg accuracy: {fmt_metric(np.nanmean(ACC_avg_per_bs), np.nanstd(ACC_avg_per_bs), ACC_avg_lo, ACC_avg_hi)} | "
            f"avg balanced accuracy: {fmt_metric(np.nanmean(BALACC_avg_per_bs), np.nanstd(BALACC_avg_per_bs), BALACC_avg_lo, BALACC_avg_hi)} | "
            f"avg specificity: {fmt_metric(np.nanmean(SPEC_avg_per_bs), np.nanstd(SPEC_avg_per_bs), SPEC_avg_lo, SPEC_avg_hi)}"
        )
        print(
            f"\t avg recall (sensitivity): {fmt_metric(np.nanmean(SENS_avg_per_bs), np.nanstd(SENS_avg_per_bs), SENS_avg_lo, SENS_avg_hi)} | "
            f"avg precision: {fmt_metric(np.nanmean(PREC_avg_per_bs), np.nanstd(PREC_avg_per_bs), PREC_avg_lo, PREC_avg_hi)} | "
            f"avg F1: {fmt_metric(np.nanmean(F1_avg_per_bs), np.nanstd(F1_avg_per_bs), F1_avg_lo, F1_avg_hi)}\n"
        )

        print('Individual AUROC:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(AUC_list[:, idx]), np.nanstd(AUC_list[:, idx]), auc_lo[idx], auc_hi[idx])}")

        print('\nIndividual AP (Average Precision, used for mAP):')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(AP_list[:, idx]), np.nanstd(AP_list[:, idx]), ap_lo[idx], ap_hi[idx])}")

        print('\nIndividual PR-AUC:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(PRAUC_list[:, idx]), np.nanstd(PRAUC_list[:, idx]), prauc_lo[idx], prauc_hi[idx])}")

        print('\nIndividual accuracy:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(accuracy_list[:, idx]), np.nanstd(accuracy_list[:, idx]), acc_lo[idx], acc_hi[idx])}")

        print('\nIndividual balanced accuracy:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(balanced_accuracy_list[:, idx]), np.nanstd(balanced_accuracy_list[:, idx]), balacc_lo[idx], balacc_hi[idx])}")

        print('\nIndividual precision:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(precision_list[:, idx]), np.nanstd(precision_list[:, idx]), prec_lo[idx], prec_hi[idx])}")

        print('\nIndividual sensitivity:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(sensitivity_list[:, idx]), np.nanstd(sensitivity_list[:, idx]), sens_lo[idx], sens_hi[idx])}")

        print('\nIndividual specificity:')
        for idx, pathology in enumerate(self.label_names):
            print(
                f"\t{pathology}: {fmt_metric(np.nanmean(specificity_list[:, idx]), np.nanstd(specificity_list[:, idx]), spec_lo[idx], spec_hi[idx])}")

        print('------------------------------------------------------'
              '----------------------------------')

        msg = (
            f'\n\n----------------------------------------------------------------------------------------\n'
            f'\t experiment:{self.params["experiment_name"]}\n\n'
            f'avg AUROC: {fmt_metric(np.nanmean(AUC_avg_per_bs), np.nanstd(AUC_avg_per_bs), AUC_avg_lo, AUC_avg_hi)} | '
            f'avg AP (mAP): {fmt_metric(np.nanmean(AP_avg_per_bs), np.nanstd(AP_avg_per_bs), AP_avg_lo, AP_avg_hi)} | '
            f'avg PR-AUC: {fmt_metric(np.nanmean(PRAUC_avg_per_bs), np.nanstd(PRAUC_avg_per_bs), PRAUC_avg_lo, PRAUC_avg_hi)}\n'
            f'avg accuracy: {fmt_metric(np.nanmean(ACC_avg_per_bs), np.nanstd(ACC_avg_per_bs), ACC_avg_lo, ACC_avg_hi)} | '
            f'avg balanced accuracy: {fmt_metric(np.nanmean(BALACC_avg_per_bs), np.nanstd(BALACC_avg_per_bs), BALACC_avg_lo, BALACC_avg_hi)} | '
            f'avg specificity: {fmt_metric(np.nanmean(SPEC_avg_per_bs), np.nanstd(SPEC_avg_per_bs), SPEC_avg_lo, SPEC_avg_hi)}\n'
            f'avg recall (sensitivity): {fmt_metric(np.nanmean(SENS_avg_per_bs), np.nanstd(SENS_avg_per_bs), SENS_avg_lo, SENS_avg_hi)} | '
            f'avg precision: {fmt_metric(np.nanmean(PREC_avg_per_bs), np.nanstd(PREC_avg_per_bs), PREC_avg_lo, PREC_avg_hi)} | '
            f'avg F1: {fmt_metric(np.nanmean(F1_avg_per_bs), np.nanstd(F1_avg_per_bs), F1_avg_lo, F1_avg_hi)}\n\n'
        )

        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)

        msg = f'Individual AUROC:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(AUC_list[:, idx]), np.nanstd(AUC_list[:, idx]), auc_lo[idx], auc_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual AP (Average Precision, used for mAP):\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(AP_list[:, idx]), np.nanstd(AP_list[:, idx]), ap_lo[idx], ap_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual PR-AUC:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(PRAUC_list[:, idx]), np.nanstd(PRAUC_list[:, idx]), prauc_lo[idx], prauc_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual accuracy:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(accuracy_list[:, idx]), np.nanstd(accuracy_list[:, idx]), acc_lo[idx], acc_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual balanced accuracy:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(balanced_accuracy_list[:, idx]), np.nanstd(balanced_accuracy_list[:, idx]), balacc_lo[idx], balacc_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual precision:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(precision_list[:, idx]), np.nanstd(precision_list[:, idx]), prec_lo[idx], prec_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual sensitivity:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(sensitivity_list[:, idx]), np.nanstd(sensitivity_list[:, idx]), sens_lo[idx], sens_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual specificity:\n'
        with open(
                os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(testsetname),
                'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {fmt_metric(np.nanmean(specificity_list[:, idx]), np.nanstd(specificity_list[:, idx]), spec_lo[idx], spec_hi[idx])} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Test_on_' + str(
                    testsetname), 'a') as f:
                f.write(msg)

        df = pd.DataFrame({
            'AUROC_mean': np.nanmean(AUC_list, axis=1),
            'AP_mAP_mean': np.nanmean(AP_list, axis=1),
            'PRAUC_mean': np.nanmean(PRAUC_list, axis=1),
            'Accuracy_mean': np.nanmean(accuracy_list, axis=1),
            'BalancedAccuracy_mean': np.nanmean(balanced_accuracy_list, axis=1),
            'Specificity_mean': np.nanmean(specificity_list, axis=1),
            'Sensitivity_mean': np.nanmean(sensitivity_list, axis=1),
            'Precision_mean': np.nanmean(precision_list, axis=1),
            'F1_mean': np.nanmean(F1_list, axis=1),
        })

        for idx in range(AUC_list.shape[-1]):
            df.insert(len(df.columns), 'AUROC_' + str(idx + 1), AUC_list[:, idx])
            df.insert(len(df.columns), 'AP_' + str(idx + 1), AP_list[:, idx])
            df.insert(len(df.columns), 'PRAUC_' + str(idx + 1), PRAUC_list[:, idx])

        df.to_csv(
            os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/bootstrapped_AUC_Test_on' + str(
                testsetname) + '.csv',
            sep=',',
            index=False
        )

        return AUC_list