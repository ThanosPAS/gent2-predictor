import torch
import torch.optim
import torch.nn as nn
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


    def __init__(self):
        super(FFN, self).__init__()

        # 1st hidden layer
        self.linear_1 = torch.nn.Linear(21920, 100)
        self.relu1 = torch.nn.ReLU()
        torch.nn.Dropout(0.2)

        # 2nd hidden layer
        self.linear_2 = torch.nn.Linear(100, 70)
        self.relu2 = torch.nn.ReLU()
        torch.nn.Dropout(0.2)

        # Output layer
        self.linear_out = torch.nn.Linear(70, 6)
        self.softmax = torch.nn.Softmax()

        # self.batchnorm1 = nn.BatchNorm1d(100)
        # self.batchnorm2 = nn.BatchNorm1d(70)

        print(self)


    def forward(self, x):
        x = self.relu1(self.linear_1(x))
        x = self.relu2(self.linear_2(x))
        x = self.linear_out(x)
        return x