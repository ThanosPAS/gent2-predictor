import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, matthews_corrcoef


class Plotter:
    def __init__(self):
        pass

    def plot_losses(self, train_loss, valid_loss, burn_in=20):
        plt.figure(figsize=(15, 4))
        plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
        plt.plot(list(range(burn_in, len(valid_loss))), valid_loss[burn_in:], label='Validation loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Minimum Validation Loss')

        plt.legend(frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


    def plot_roc_curve(self, test, pred):
        #Define where pred comes from
        #I am not sure if the threshold should be 50%
        y_test_class = np.where(y_test.flatten() >= 0.5, 1, 0)
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
        plt.show()

        return y_test_class, y_pred_class



    def plot_mcc(self):
        plt.title('Matthews Correlation Coefficient')
        plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label='MCC = %0.2f' % mcc)
        plt.legend(loc='lower right')
        plt.ylabel('Predicted')
        plt.xlabel('Validation targets')
        plt.show()


    def stats(self):
        sns.countplot(x='Cancer types', data=output_classes)

    def accuracy(self):
        accuracy_stats['train'].append(train_epoch_acc / len(x_train))
        accuracy_stats['val'].append(val_epoch_acc / len(x_val))
        print(
            f'Epoch {e + 0:03}: | Train Acc: {train_epoch_acc / len(x_train):.3f}| Val Acc: {val_epoch_acc / len(x_val):.3f}')

    def stat_significance():
        mcc = matthews_corrcoef(y_test_class, y_pred_class)
        raise NotImplementedError
