import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


unpickling()


def data_loading():
    #Training data
    x_train = DataLoader(sth, )
    y_train = DataLoader(sth, )
    #Validating the models performance
    x_val = DataLoader(sth, )
    y_val = DataLoader(sth, )
    #Never seen data by the model
    x_test = DataLoader(sth, )
    y_test = DataLoader(sth, )
    return x_train, y_train, x_val, y_val, x_test, y_test


# Model structure
class FFN(nn.Module):
    r"""An artificial neural network (ANN) for predicting

    Parameters
    ----------
    n_features : int
        Number of patients in input. Default = 1
    num_classes : list
        List of classes with each element being either a group cancer subtype patients or a cohort of normal people. This is also the number of units in the output layer. Default predifined list.
    p_dropout : float
        Probability of dropout, by default 0.2
    n_hidden1-x :
        Number of hidden units in the 1-x layers. Default: x


    Examples
    ----------
        >>> net = ANN(in_features=10, p_dropout = 0.2)
        >>> print(net)

    """


    def __init__(self, n_features=1, num_classes=7, drop_out=0.2, num_hidden_1=16, num_hidden_2=16):
        super(Net, self).__init__()

        # 1st hidden layer
        self.linear_1 = torch.nn.Linear(n_features, num_hidden_1)
        self.relu1 = torch.nn.ReLU()
        torch.nn.Dropout(drop_out)

        # 2nd hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.relu2 = torch.nn.ReLU()
        torch.nn.Dropout(drop_out)

        # Output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)

        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        out = self.linear_l(x)
        out = F.relu1(out)
        out = self.linear_2(out)
        out = F.relu2(out)
        logits = self.linear_out(out)
        probas = self.softmax(logits, dim=1)
        return logits, probas


#How do I connect the parsing arguments in __main__ with the functions in this file?

# Optimization
if OPTIMIZER.upper() == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_reg)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)


# Train

for epoch in range(num_epochs):
    model.train()
for batch_idx, (features, targets) in enumerate(train_loader):
    features = features.view(-1, 28 * 28).to(device)
    targets = targets.to(device)

criterion = F.cross_entropy(sth, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
# compute accuracy






def ffn_predict()


