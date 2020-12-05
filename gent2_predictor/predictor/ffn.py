import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
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
        self.dropout = nn.Dropout(0.2)
        # 1st hidden layer
        self.linear_1 = nn.Linear(21920, 100)
        self.batchnorm1 = nn.BatchNorm1d(100,momentum=0.1)
        self.relu1 = nn.ReLU()
        #self.dropout = nn.Dropout(0.2)

        # 2nd hidden layer
        self.linear_2 = nn.Linear(100, 70)
        self.batchnorm2 = nn.BatchNorm1d(70,momentum=0.1)
        self.relu2 = nn.ReLU()


        # Output layer
        self.linear_out = nn.Linear(70, 6)
        self.softmax = nn.Softmax()

    '''
    def backbone_ffn(self):
        nn.Sequential(
        nn.Linear(21920, 100),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(100, 70),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(70, 6)
    )
        x =self.backbone_fnn(x)
        self.layer_grid = {
            'Linear_1': nn.Linear(21920, 100),
            'Dropout_1': nn.Dropout(0.2),
            'RelU_1': nn.ReLU(),
            'Linear_2': nn.Linear(100, 70),
            'Dropout_2':nn.Dropout(0.2),
            'RelU_2': nn.ReLU(),
            'Linear_out':nn.Linear(70, 6)
       }
        
        #print(self)


        '''
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        '''
        #x = self.batchnorm0(x)
        x = self.batchnorm1(self.linear_1(x))
        nn.Dropout(0.2)
        self.relu1(x)
        x = self.batchnorm2(self.linear_2(x))
        
        self.relu2(x)
        self.linear_out(x)
        '''
        x = self.relu1(self.linear_1(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu2(self.linear_2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = self.linear_out(x)
        #x = self.softmax(x)


        
        #x = nn.Sequential(self.layer_grid.items())'''
        return x

