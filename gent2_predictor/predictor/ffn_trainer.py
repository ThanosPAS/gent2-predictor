import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.predictor.plotter import Plotter
from gent2_predictor.predictor.trainer import Trainer
from gent2_predictor.settings import DEVICE, OPTIMIZER, LEARNING_RATE, L2_REG, MOMENTUM, \
    INIT_METHOD, USE_CUDA, EPOCHS, MODEL_SELECTOR


class FFNTrainer(Trainer):
    def __init__(self, model=None, full_data=False):
        super().__init__()
        self.model = model
        self.model.to(DEVICE)
        self.model_name = ''
        # self.model.apply(self.init_weights)

        self.criterion = nn.CrossEntropyLoss()

        if OPTIMIZER.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)

        self.train_loader, self.val_loader, self.test_loader = DataParser().data_loading(full_data)

    @staticmethod
    def init_weights(m):
        # FIXME: Correct the method to be used by model.apply()
        init_function = getattr(nn.init, INIT_METHOD, None)
        init_function(m.weight)
        nn.init.constant_(m.bias, 0)

    def start_loop(self):
        print('Training\n')
        if USE_CUDA:
            self.model.cuda()
            long_tensor = torch.cuda.LongTensor
            float_tensor = torch.cuda.FloatTensor
        else:
            long_tensor = torch.LongTensor
            float_tensor = torch.FloatTensor

        train_loss, valid_loss = [], []
        train_epoch_acc, val_epoch_acc = dict(), dict()

        for epoch in range(EPOCHS):
            self.model.train()
            train_batch_loss = 0
            val_batch_loss = 0
            train_epoch_acc[epoch] = 0
            val_epoch_acc[epoch] = 0
            running_train_acc, running_val_acc = [], []

            with tqdm(total=len(self.train_loader.dataset),
                      desc=f"[Epoch {epoch + 1:3d}/{EPOCHS}]") as pbar:

                for idx_batch, person in enumerate(self.train_loader):

                    x_train = person['data'].type(float_tensor)
                    y_train = person['cancer_type'].type(long_tensor)

                    pred = self.model(x_train)
                    t_loss = self.criterion(pred, y_train)
                    personal_train_acc = self.multi_acc(pred, y_train)
                    running_train_acc.append(personal_train_acc)

                    self.optimizer.zero_grad()
                    t_loss.backward()
                    self.optimizer.step()

                    train_batch_loss += t_loss.item()

                    pbar.set_postfix({'loss': train_batch_loss})
                    pbar.update(x_train.shape[0])

                trainset_acc = sum(running_train_acc) / len(running_train_acc)
                trainset_acc = round(trainset_acc, 3) * 100
                train_epoch_acc[epoch] += trainset_acc

                train_loss.append(train_batch_loss / len(self.train_loader))
                with torch.no_grad():

                    self.model.eval()

                    for person in self.val_loader:

                        x_val = person['data'].type(float_tensor)
                        y_val = person['cancer_type'].type(long_tensor)

                        pred = self.model(x_val)
                        v_loss = self.criterion(pred, y_val)
                        personal_valset_acc = self.multi_acc(pred, y_val)
                        running_val_acc.append(personal_valset_acc)

                        val_batch_loss += v_loss.item()

                    valset_acc = sum(running_val_acc) / len(running_val_acc)
                    valset_acc = round(valset_acc, 3) * 100
                    val_epoch_acc[epoch] += valset_acc

                    valid_loss.append(val_batch_loss / len(self.val_loader))

                pbar.set_postfix({
                    'loss'    : train_loss[epoch],
                    'val_loss': valid_loss[epoch],
                    'acc'     : train_epoch_acc[epoch],
                    'val_acc' : val_epoch_acc[epoch]
                })
        if MODEL_SELECTOR == 'FULL_FFN':
            self.model_name = self.save_model(self.model, 'ffn')
        else:
            self.model_name = self.save_model(self.model, 'baselineFFN')

        plotter = Plotter(self.model_name)
        plotter.plot_losses(train_loss, valid_loss)
        self.save_predictions(self.model_name, loss_list=None, train_loss=train_loss,
                              valid_loss=valid_loss, mode=False)

        return train_loss, valid_loss, train_epoch_acc, val_epoch_acc

    def multi_acc(self, pred, y):
        y_pred_softmax = torch.log_softmax(pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y)
        if correct_pred:
            acc = 1
        else:
            acc = 0

        return acc

    def predict(self, model_filename):
        print('Predicting\n')
        self.model_name = model_filename

        if USE_CUDA:
            self.model.cuda()
            long_tensor = torch.cuda.LongTensor
            float_tensor = torch.cuda.FloatTensor
        else:
            long_tensor = torch.LongTensor
            float_tensor = torch.FloatTensor

        test_batch_loss = 0
        test_loss = 0
        pred_labels, loss_list, running_test_acc, y_test_list = [], [], [], []

        with torch.no_grad():
            self.model.eval()
            i = 0
            for person in self.test_loader:
                with tqdm(total=len(self.test_loader.dataset),
                          desc=f"[person {i + 1:3d}/{len(self.test_loader.dataset)}]") as pbar:
                    x_test = person['data'].type(float_tensor)
                    y_test = person['cancer_type'].type(long_tensor)
                    pred = self.model(x_test)

                    t_loss = self.criterion(pred, y_test)
                    personal_test_acc = self.multi_acc(pred, y_test)
                    running_test_acc.append(personal_test_acc)
                    test_batch_loss += t_loss.item()
                    loss_cast = t_loss.tolist()
                    loss_str = str(loss_cast)
                    loss_list.append(loss_str)
                    y_test_list.append(y_test.item())
                    pbar.update(x_test.shape[0])
                    y_pred_softmax = torch.log_softmax(pred, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    pred_labels.append(y_pred_tags.item())
                    # print('Predicted cancer type for patient ', patient[person], 'is: ', pred_labels[person])
                    testset_acc = sum(running_test_acc) / len(running_test_acc)
                    testset_acc = round(testset_acc, 3) * 100
                    pbar.set_postfix({
                        'loss'                : t_loss.item(),
                        'accumulated_test_acc': testset_acc
                    })

                i += 1
            test_loss = test_batch_loss / len(self.val_loader)
            test_loss = round(test_loss, 2) * 100
            pred_arr = np.asarray(pred_labels)
            y_test_arr = np.asarray(y_test_list)

            self.save_predictions(self.model_name,loss_list, train_loss=None, valid_loss=None, mode=True)

            plotter = Plotter(self.model_name)
            plotter.plot_cm(y_test_arr, pred_arr)
            #plotter.plot_roc_curve(y_test_arr,pred_arr)

        print('Prediction successful')
        print('Overall test accuracy:', testset_acc, 'Overall test loss:', test_loss)
