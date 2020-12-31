import warnings

import torch.nn as nn

warnings.filterwarnings("ignore")


# Model structure
class FFN(nn.Module):

    def __init__(self):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.layernorm0 = nn.LayerNorm(21920)
        # 1st hidden layer
        self.linear_1 = nn.Linear(21920, 16000)
        self.layernorm1 = nn.LayerNorm(16000)
        self.relu1 = nn.ReLU()


        # 2nd hidden layer
        self.linear_2 = nn.Linear(16000, 10000)
        self.layernorm2 = nn.LayerNorm(10000)
        self.relu2 = nn.ReLU()
        # 3rd hidden layer
        self.linear_3 = nn.Linear(10000, 6000)
        self.layernorm3 = nn.LayerNorm(6000)
        self.relu3 = nn.ReLU()
        # 4th hidden layer
        self.linear_4 = nn.Linear(6000, 3000)
        self.layernorm4 = nn.LayerNorm(3000)
        self.relu4 = nn.ReLU()
        # 5th hidden layer
        self.linear_5 = nn.Linear(3000, 1000)
        self.layernorm5 = nn.LayerNorm(1000)
        self.relu5 = nn.ReLU()
        # 6th hidden layer
        self.linear_6 = nn.Linear(1000, 100)
        self.layernorm6 = nn.LayerNorm(100)
        self.relu6 = nn.ReLU()
        # Output layer
        self.linear_out = nn.Linear(100, 8)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layernorm0(x)

        x = self.relu1(self.linear_1(x))
        x = self.layernorm1(x)
        x = self.dropout(x)


        x = self.relu2(self.linear_2(x))
        x = self.layernorm2(x)
        x = self.dropout(x)


        x = self.relu3(self.linear_3(x))
        x = self.layernorm3(x)
        x = self.dropout(x)


        x = self.relu4(self.linear_4(x))
        x = self.layernorm4(x)
        x = self.dropout(x)


        x = self.relu5(self.linear_5(x))
        x = self.layernorm5(x)
        x = self.dropout(x)

        x = self.relu6(self.linear_6(x))
        x = self.layernorm6(x)
        x = self.dropout(x)

        x = self.linear_out(x)

        return x


class Baseline_FFN(nn.Module):
    def __init__(self):
        super(Baseline_FFN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.layernorm0 = nn.LayerNorm(21920)
        # 1st hidden layer
        self.linear_1 = nn.Linear(21920, 5000)
        self.layernorm1 = nn.LayerNorm(5000)
        self.relu1 = nn.ReLU()

        # 2nd hidden layer
        self.linear_2 = nn.Linear(5000, 500)
        self.layernorm2 = nn.LayerNorm(500)
        self.relu2 = nn.ReLU()

        # Output layer
        self.linear_out = nn.Linear(500, 8)

    def forward(self, x):

        x = self.layernorm0(x)

        x = self.relu1(self.linear_1(x))

        x = self.relu2(self.linear_2(x))

        x = self.linear_out(x)

        return x

class Landmarks_full(nn.Module):

    def __init__(self):
        super(Landmarks_full, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.layernorm0 = nn.LayerNorm(968)
        # 1st hidden layer
        self.linear_1 = nn.Linear(968, 700)
        self.layernorm1 = nn.LayerNorm(700)
        self.relu1 = nn.ReLU()


        # 2nd hidden layer
        self.linear_2 = nn.Linear(700, 500)
        self.layernorm2 = nn.LayerNorm(500)
        self.relu2 = nn.ReLU()
        # 3rd hidden layer
        self.linear_3 = nn.Linear(500, 300)
        self.layernorm3 = nn.LayerNorm(300)
        self.relu3 = nn.ReLU()
        # 4th hidden layer
        self.linear_4 = nn.Linear(300, 200)
        self.layernorm4 = nn.LayerNorm(200)
        self.relu4 = nn.ReLU()
        # 5th hidden layer
        self.linear_5 = nn.Linear(200, 100)
        self.layernorm5 = nn.LayerNorm(100)
        self.relu5 = nn.ReLU()
        # Output layer
        self.linear_out = nn.Linear(100, 8)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layernorm0(x)

        x = self.relu1(self.linear_1(x))
        x = self.layernorm1(x)
        x = self.dropout(x)


        x = self.relu2(self.linear_2(x))
        x = self.layernorm2(x)
        x = self.dropout(x)


        x = self.relu3(self.linear_3(x))
        x = self.layernorm3(x)
        x = self.dropout(x)


        x = self.relu4(self.linear_4(x))
        x = self.layernorm4(x)
        x = self.dropout(x)


        x = self.relu5(self.linear_5(x))
        x = self.layernorm5(x)
        x = self.dropout(x)

        x = self.linear_out(x)

        return x


class Landmarks_baseline(nn.Module):
    def __init__(self):
        super(Landmarks_baseline, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.layernorm0 = nn.LayerNorm(968)
        # 1st hidden layer
        self.linear_1 = nn.Linear(968, 500)
        self.layernorm1 = nn.LayerNorm(500)
        self.relu1 = nn.ReLU()

        # 2nd hidden layer
        self.linear_2 = nn.Linear(500, 100)
        self.layernorm2 = nn.LayerNorm(100)
        self.relu2 = nn.ReLU()

        # Output layer
        self.linear_out = nn.Linear(100, 8)

    def forward(self, x):

        x = self.layernorm0(x)

        x = self.relu1(self.linear_1(x))

        x = self.relu2(self.linear_2(x))

        x = self.linear_out(x)

        return x