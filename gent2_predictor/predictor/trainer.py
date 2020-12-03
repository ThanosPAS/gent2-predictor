import os

from gent2_predictor.data_parser.data_parser import DataParser

from gent2_predictor.settings import DEVICE, OPTIMIZER, LEARNING_RATE, L2_REG, MOMENTUM, \
    INIT_METHOD, USE_CUDA, EPOCHS, MODEL_PATH, MODEL_PATH_DIR
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model=None):
        self.model = model
        self.model.to(DEVICE)
        # self.model.apply(self.init_weights)

        self.criterion = nn.CrossEntropyLoss()

        if OPTIMIZER.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)
        self.train_loader, self.val_loader, self.test_loader = DataParser().data_loading()


    @staticmethod
    def init_weights(m):
        # FIXME: Correct the method to be used by model.apply()
        init_function = getattr(nn.init, INIT_METHOD, None)
        init_function(m.weight)
        nn.init.constant_(m.bias, 0)

    def start_loop(self):
        print('Training')
        if USE_CUDA:
            self.model.cuda()
            LongTensor = torch.cuda.LongTensor
            FloatTensor = torch.cuda.FloatTensor
        else:
            LongTensor = torch.LongTensor
            FloatTensor = torch.FloatTensor

        train_loss, valid_loss = [], []
        train_epoch_acc, val_epoch_acc = dict(), dict()

        for epoch in range(EPOCHS):
            print(f'Epoch: {epoch}')
            self.model.train()
            batch_loss = 0
            train_epoch_acc[epoch] = 0
            val_epoch_acc[epoch] = 0

            for person in self.train_loader:
                x_train = person['data']
                y_train = person['cancer_type']
                y_train = y_train.type(LongTensor)
                x_train = x_train.type(FloatTensor)
                pred = self.model(x_train)
                loss = self.criterion(pred, y_train)
                train_acc = self.multi_acc(pred, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.data
                train_epoch_acc[epoch] += train_acc.cpu().numpy()

                train_loss.append(batch_loss / len(x_train))

            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))

            with torch.no_grad():
                self.model.eval()
                batch_loss = 0
                for person in self.val_loader:
                    x_val = person['data']
                    y_val = person['cancer_type']
                    y_val = y_val.type(LongTensor)
                    x_val = x_val.type(FloatTensor)
                    pred = self.model(x_val)
                    y_pred_softmax = torch.log_softmax(pred, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    loss = self.criterion(pred, y_val)
                    val_acc = self.multi_acc(pred, y_val)
                    batch_loss += loss.data
                    val_epoch_acc[epoch] += val_acc.cpu().numpy()

                valid_loss.append(loss.data)

        self.save_model()
        return train_loss, valid_loss, train_epoch_acc, val_epoch_acc

    def multi_acc(self, val_pred, y_val):
        y_pred_softmax = torch.log_softmax(val_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_val).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc) * 100

        return acc

    def predict(self):
        y_pred_list = []
        test_loss = []
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

    def save_model(self):
        if not os.path.exists(MODEL_PATH_DIR):
            os.makedirs(MODEL_PATH_DIR)

        torch.save(self.model.state_dict(), MODEL_PATH)
