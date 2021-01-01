import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from gent2_predictor.settings import PLOTS_PATH_DIR, TARGET_LABELS, PREDICTIONS_PATH_DIR


class Plotter:
    def __init__(self, model_name):
        self.model_name = model_name

    def plot_losses(self, train_loss, valid_loss, burn_in=1):
        sns.set_theme()
        sns.set_palette('icefire')
        plt.figure(figsize=(15, 8))
        plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:],
                 label='Training loss')
        plt.plot(list(range(burn_in, len(valid_loss))), valid_loss[burn_in:],
                 label='Validation loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Minimum Validation Loss')

        plt.legend(frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'losses_{self.model_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    def accuracy(self, train_acc_list, val_acc_list, test_acc_list=None,burn_in=1, mode=True):
        if mode:
            sns.set_theme()
            sns.set_palette('icefire')
            plt.figure(figsize=(15, 8))
            plt.plot(list(range(burn_in, len(train_acc_list))), train_acc_list[burn_in:],
                     label='Train accuracy')
            plt.plot(list(range(burn_in, len(val_acc_list))), val_acc_list[burn_in:],
                     label='Validation accuracy')
            # find position of lowest validation loss
            minposs = val_acc_list.index(min(val_acc_list)) + 1
            plt.axvline(minposs, linestyle='--', color='r', label='Minimum Validation Accuracy')
        else:
            sns.set_theme()
            sns.set_palette('icefire')
            plt.figure(figsize=(15, 8))
            plt.plot(list(range(burn_in, len(test_acc_list))), test_acc_list[burn_in:],
                     label='Test accuracy')
            # find position of lowest validation loss
            minposs = test_acc_list.index(min(test_acc_list)) + 1
            plt.axvline(minposs, linestyle='--', color='r', label='Minimum Test Accuracy')

        plt.legend(frameon=False)
        plt.xlabel('Patients')
        plt.ylabel('Accuracy')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'accuracy_{self.model_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    def plot_roc_curve(self, test, pred):
        # Define where pred comes from
        # I am not sure if the threshold should be 50%
        y_test_class = np.where(test.flatten() >= 0.5, 1, 0)
        y_pred_class = np.where(pred.flatten() >= 0.5, 1, 0)

        # ### Receiver Operating Caracteristic (ROC) curve

        fpr, tpr, threshold = roc_curve(y_test_class, pred.flatten().detach().numpy())
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'roc_{self.model_name}.png')
        plt.savefig(plot_path)
        plt.show()

        return y_test_class, y_pred_class

    def plot_mcc(self, test, pred, mcc):
        plt.title('Matthews Correlation Coefficient')
        plt.scatter(test.flatten().detach().numpy(), pred.flatten().detach().numpy(),
                    label='MCC = %0.2f' % mcc)
        plt.legend(loc='lower right')
        plt.ylabel('Predicted')
        plt.xlabel('Validation targets')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'mcc_{self.model_name}.png')
        plt.savefig(plot_path)
        plt.show()

    # def stats(self):
    #     sns.countplot(x='Cancer types', data=output_classes)
    #
    # def accuracy(self):
    #     accuracy_stats['train'].append(train_epoch_acc / len(x_train))
    #     accuracy_stats['val'].append(val_epoch_acc / len(x_val))
    #     print(
    #         f'Epoch {e + 0:03}: | Train Acc: {train_epoch_acc / len(x_train):.3f}| Val Acc: {val_epoch_acc / len(x_val):.3f}')

    def stat_significance(y, pred):
        mcc = matthews_corrcoef(y, pred)
        raise NotImplementedError

    def plot_cm(self, y_test_arr, pred_arr, figsize=(5, 5)):
        cm = confusion_matrix(y_test_arr, pred_arr, labels=np.unique(y_test_arr))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=TARGET_LABELS.keys(), columns=TARGET_LABELS.keys())
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, cmap="PuRd", annot=annot, fmt='', ax=ax)

        plot_path = os.path.join(PLOTS_PATH_DIR, f'cm_{self.model_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    def plot_init_methods(self):
        sns.set_theme()
        sns.set_palette('gnuplot2_r')

        init_methods = {
            'Xavier – uniform' : 'train-val_losses_ffn_2020-12-13_14.36.32.pth_2020-12-13_14.36.48(xavier_iniform_15-15).csv',
            'Xavier – normal'  : 'train-val_losses_ffn_2020-12-13_14.41.37.pth_2020-12-13_14.41.53(xavier_normal_15-15).csv',
            'Kaiming - uniform': 'train-val_losses_ffn_2020-12-13_14.45.39.pth_2020-12-13_14.45.54(kaiming_unifirm_15-15).csv',
            'Kaiming - normal' : 'train-val_losses_ffn_2020-12-13_14.53.55.pth_2020-12-13_14.54.11(kaiming_normal_15-15).csv',
            'Orthogonal'       : 'train-val_losses_ffn_2020-12-13_14.59.21.pth_2020-12-13_14.59.37(orthogonal_15-15).csv',
            'Sparse'           : 'train-val_losses_ffn_2020-12-13_15.02.51.pth_2020-12-13_15.03.08(sparse_15-15).csv',
        }

        plt.figure(figsize=(15, 8))

        for key, value in init_methods.items():
            path = os.path.join(PREDICTIONS_PATH_DIR, value)
            df = pd.read_csv(path)
            plt.plot(list(range(10)), df['validation_loss'].iloc[:10], label=key)

        plt.legend(frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'losses_{self.model_name}_init_methods.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

# burnin = 30
#
# sns.set_theme()
# sns.set_palette('Accent')
# plt.figure(figsize=(8, 8))
#
# plt.plot(list(range(burnin, len(baseline_t))), baseline_t[burnin:], label='Baseline training loss')
# plt.plot(list(range(burnin, len(baseline_v))), baseline_v[burnin:], label='Baseline validation loss')
#
# plt.plot(list(range(burnin, len(normal_t))), normal_t[burnin:], label='Final training loss')
# plt.plot(list(range(burnin, len(normal_v))), normal_v[burnin:], label='Final validation loss')
#
# # find position of lowest validation loss
# minposs = baseline_v.index(min(baseline_v)) + 1
# plt.axvline(minposs, linestyle='--', color='r', label='Minimum baseline validation Loss')
#
# minposs = normal_v.index(min(normal_v)) + 1
# plt.axvline(minposs, linestyle='--', color='r', label='Minimum final validation Loss')
#
# plt.legend(frameon=False)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plot_path = os.path.join(PLOTS_PATH_DIR, f'losses_500.pdf')
# plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
# plt.show()
