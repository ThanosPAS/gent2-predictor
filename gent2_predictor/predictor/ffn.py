import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.settings import EPOCHS, LEARNING_RATE, OPTIMIZER, INIT_METHOD, L2_REG, \
    MOMENTUM, USE_CUDA, DEVICE

import warnings
warnings.filterwarnings("ignore")
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
        self.train_loader, self.val_loader, self.test_loader = DataParser().data_loading()
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

        print(self)


    def forward(self, x):
        x = self.linear_l(x)
        x = self.batchnorm1(x)
        x = F.relu1(x)

        x = self.linear_2(x)
        x = self.batchnorm2(x)
        x = F.relu2(x)
        return x

        #logits = self.linear_out(x)
        #probas = self.softmax(logits, dim=1)
        #return logits, probas


class FFNTrainer:
    def __init__(self, model=None):
        if not model:
            self.model = FFN()
            self.model.to(DEVICE)
            self.model.apply(self.init_weights)

        self.criterion = F.cross_entropy

        if OPTIMIZER.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)

    @staticmethod
    def init_weights(m):
        init_function = getattr(nn.init, INIT_METHOD, None)
        if isinstance(INIT_METHOD, nn.init):
            init_function(m.weight)
            nn.init.constant_(m.bias, 0)

    def train_ffn(self):
        if USE_CUDA:
            self.model.cuda()



        train_loss, valid_loss, y_pred_list, output_classes = [], [], [], []

        for epoch in range(EPOCHS):
            self.model.train()
            #If we want to have control of how many patients we want to have as an input.
            #How the for loop that follow will be written it will depend on what structure the x_train has
            batch_loss = 0
            train_epoch_acc = 0
            val_epoch_acc = 0

            for person in self.train_loader:
                x_train = person['data']
                y_train = person['cancer_type']
                #features = features.view(-1, 28 * 28).to(device)
                pred = self.model(x_train)
                loss = self.criterion(pred, y_train)
                train_acc = self.multi_acc(pred, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.data
                train_epoch_acc += train_acc.item()

            train_loss.append(batch_loss / len(x_train))

            if epoch % (EPOCHS // 10) == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))

            with torch.no_grad():
                self.model.eval()
                batch_loss = 0
                for person in self.val_loader:
                    x_val = person['data']
                    y_val = person['cancer_type']

                    pred = self.model(person)
                    loss = self.criterion(pred, y_val)
                    val_acc = self.multi_acc(pred, y_val)
                    batch_loss += loss.data
                    val_epoch_acc += val_acc.item()
                    ##Returns the cancer type - not sure yet how to do it
                    y_pred_softmax = torch.log_softmax(pred, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    y_pred_list.append(y_pred_tags.cpu().numpy())
                    output_classes.append(cancer_types[])

                valid_loss.append(loss.data)

            return train_loss, valid_loss, train_epoch_acc, val_epoch_acc, output_classes


    def multi_acc(self, val_pred, y_val):
        y_pred_softmax = torch.log_softmax(val_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_val).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc) * 100

        return acc

    def predict_ffn(self):
        y_pred_list = []
        test_loss = []
        test_label = ''
        with torch.no_grad():
            self.model.eval()
            batch_loss = 0
            for person in self.test_loader:
                y_test = person['cancer_type']
                pred = self.model(person)
                loss = self.criterion(pred, y_test)
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
