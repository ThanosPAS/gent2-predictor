import numpy as np
import seaborn as sns
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from gent2_predictor.data_parser.data_parser import unpickling

# switch to GPU if given
use_cuda=torch.cuda.is_available()


cancer_types = ['Normal', 'Breast', 'Colon','Gastric','Lung', 'Liver', ]

# Model structure
class FFN(nn.Module):
    r"""An artificial neural network (ANN) for predicting

    Parameters
    ----------
    n_patients : int
        Number of patients in input. Default = 1
    num_classes : list
        List of classes with each element being either a group cancer subtype patients or a cohort of normal people. This is also the number of units in the output layer. Default predifined list.
    drop_out : float
        Probability of dropout, by default 0.2
    n_hidden1-x :
        Number of hidden units in the 1-x layers. Default: x


    Examples
    ----------
        >>> net = ANN(in_features=10, p_dropout = 0.2)
        >>> print(net)

    """


    def __init__(self, n_patients=1, num_classes=6, drop_out=0.2, num_hidden_1=16, num_hidden_2=16):
        super(FFN, self).__init__()

        # 1st hidden layer
        self.linear_1 = torch.nn.Linear(n_patients, num_hidden_1)
        self.relu1 = torch.nn.ReLU()
        torch.nn.Dropout(drop_out)

        # 2nd hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.relu2 = torch.nn.ReLU()
        torch.nn.Dropout(drop_out)

        # Output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.softmax = torch.nn.Softmax()

        self.batchnorm1 = nn.BatchNorm1d(num_hidden_1)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden_2)


    def forward(self, x):
        x= self.linear_l(x)
        x= self.batchnorm1(x)
        x= F.relu1(x)

        x= self.linear_2(x)
        x= self.batchnorm2(x)
        x= F.relu2(x)
        return x

        #logits = self.linear_out(x)
        #probas = self.softmax(logits, dim=1)
        #return logits, probas

print(model)
#How do I connect the parsing arguments in __main__ with the functions in this file?

# Optimization
if OPTIMIZER.upper() == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)

#Loss function
criterion = F.cross_entropy()


# Train

def train_ffn(use_cuda = True):
    if use_cuda:
        model.cuda()
    train_loss, valid_loss, y_pred_list, output_classes = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        #If we want to have control of how many patients we want to have as an input.
        #How the for loop that follow will be written it will depend on what structure the x_train has
        batch_loss = 0
        train_epoch_acc = 0
        val_epoch_acc = 0
        for person in x_train:
            #features = features.view(-1, 28 * 28).to(device)
            pred = model(person)
            loss = criterion(pred, y_train)
            train_acc = multi_acc(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.data
            train_epoch_acc += train_acc.item()
        train_loss.append(batch_loss / len(x_train))
        if epoch % (EPOCHS // 10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))
        with torch.no_grad():
            model.eval()
            batch_loss = 0
            for person in x_val:
                pred = model(person)
                loss = criterion(pred, y_val)
                val_acc = multi_acc(pred, y_val)
                batch_loss += loss.data
                val_epoch_acc += val_acc.item()
                ##Returns the cancer type - not sure yet how to do it
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
                output_classes.append(cancer_types[])
            valid_loss.append(loss.data)



        return model, train_loss, valid_loss, train_epoch_acc, val_epoch_acc, output_classes


def multi_acc(val_pred, y_val):
    y_pred_softmax = torch.log_softmax(val_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc

accuracy_stats = {
    'train': [],
    "val": [],
    'test': []
}

model, train_loss, valid_loss, train_epoch_acc, val_epoch_acc, output_classes = train_ffn()



def ffn_predict():
    y_pred_list = []
    test_loss = []
    test_label = ''
    with torch.no_grad():
        model.eval()
        batch_loss = 0
        for person in x_test:
            pred = model(person)
            loss = criterion(pred, y_test)
            #Extracting the label that has the biggest probability
            y_pred_softmax = torch.log_softmax(pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        test_loss.append(loss.data)
        #Returns the cancer type - not sure yet how to do it
        test_label = 'sth'
        print(test_label)
    return test_loss, test_label

ffn_predict()






class Plotting():

    def plot_losses(burn_in=20):
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


    def plot_roc_curve(test,pred):
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



    def plot_mcc():
        plt.title('Matthews Correlation Coefficient')
        plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label='MCC = %0.2f' % mcc)
        plt.legend(loc='lower right')
        plt.ylabel('Predicted')
        plt.xlabel('Validation targets')
        plt.show()


    def stats():
        sns.countplot(x='Cancer types', data=output_classes)

    def accuracy():
        accuracy_stats['train'].append(train_epoch_acc / len(x_train))
        accuracy_stats['val'].append(val_epoch_acc / len(x_val))
        print(
            f'Epoch {e + 0:03}: | Train Acc: {train_epoch_acc / len(x_train):.3f}| Val Acc: {val_epoch_acc / len(x_val):.3f}')

    def stat_significance():


Plotting()

# ### Matthew's Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test_class, y_pred_class)






