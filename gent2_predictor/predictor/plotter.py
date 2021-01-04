import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from gent2_predictor.settings import PLOTS_PATH_DIR, TARGET_LABELS, PREDICTIONS_PATH_DIR, DATA_DIR


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
                     label='Accumulated Train Accuracy')
            plt.plot(list(range(burn_in, len(val_acc_list))), val_acc_list[burn_in:],
                     label='Accumulated Validation Accuracy')
            # find position of lowest validation loss
            minposs = val_acc_list.index(min(val_acc_list)) + 1
            plt.axvline(minposs, linestyle='--', color='r', label='Minimum Accumulated Validation Accuracy')
        else:
            sns.set_theme()
            sns.set_palette('icefire')
            plt.figure(figsize=(15, 8))
            plt.plot(list(range(burn_in, len(test_acc_list))), test_acc_list[burn_in:],
                     label='Accumulated Test Accuracy')
            # find position of lowest validation loss
            minposs = test_acc_list.index(min(test_acc_list)) + 1
            plt.axvline(minposs, linestyle='--', color='r', label='Minimum Accumulated Test Accuracy')

        plt.legend(frameon=False)
        plt.xlabel('Patients')
        plt.ylabel('Accumulated Accuracy')
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

    def stat_significance(self, y, pred):
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

    def plot_train(self):
        EPOCHS = 100
        train = {
            'full_train' : pd.read_csv(os.path.join(DATA_DIR, 'losses', 'full_train.csv')),
            'landmarks_train' : pd.read_csv(os.path.join(DATA_DIR, 'losses', 'landmarks_train.csv')),
            'full_base_train' : pd.read_csv(os.path.join(DATA_DIR, 'losses', 'full_base_train.csv')),
            'landmarks_base_train' : pd.read_csv(os.path.join(DATA_DIR, 'losses', 'landmarks_base_train.csv')),
        }

        plt.figure(figsize=(8, 8))

        c_t_loss = ['#00e63d', '#00bbe6', '#f23d3d', '#ce8eed']
        c_v_loss = ['#348a4b', '#368091', '#a11818', '#9b00e8']

        for index, key in enumerate(train.items()):
            key, df = key
            plt.plot(list(range(EPOCHS)), df['train_loss'].iloc[:EPOCHS], label=f'training_{key}', color=c_t_loss[index])
            plt.plot(list(range(EPOCHS)), df['validation_loss'].iloc[:EPOCHS], label=f'validation_{key}', color=c_v_loss[index])

        plt.legend(frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')
        plot_path = os.path.join(PLOTS_PATH_DIR, f'losses_train.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

        plt.figure(figsize=(8, 8))
        f, ax = plt.subplots(1)

        for index, key in enumerate(train.items()):
            key, df = key
            plt.plot(list(range(EPOCHS)), df['train_accuracy'].iloc[:EPOCHS], label=f'training_{key}', color=c_t_loss[index])
            plt.plot(list(range(EPOCHS)), df['validation_accuracy'].iloc[:EPOCHS], label=f'validation_{key}', color=c_v_loss[index])

        plt.legend(frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Training accuracy')
        ax.set_ylim(bottom=65)
        plot_path = os.path.join(PLOTS_PATH_DIR, f'acc_train.pdf')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()
