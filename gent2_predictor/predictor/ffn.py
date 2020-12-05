import warnings

import torch.nn as nn

warnings.filterwarnings("ignore")


# Model structure
class FFN(nn.Module):

    def __init__(self):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        # 1st hidden layer
        self.linear_1 = nn.Linear(21920, 100)
        self.batchnorm1 = nn.BatchNorm1d(100, momentum=0.1)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

        # 2nd hidden layer
        self.linear_2 = nn.Linear(100, 70)
        self.batchnorm2 = nn.BatchNorm1d(70, momentum=0.1)
        self.relu2 = nn.ReLU()

        # Output layer
        self.linear_out = nn.Linear(70, 6)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.linear_1(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu2(self.linear_2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = self.linear_out(x)

        return x
