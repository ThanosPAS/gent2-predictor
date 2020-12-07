import os
import os.path
import torch
import torch.nn as nn
from tqdm import tqdm

from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.settings import DEVICE, OPTIMIZER, LEARNING_RATE, L2_REG, MOMENTUM, \
    INIT_METHOD, USE_CUDA, EPOCHS, MODEL_PATH, MODEL_PATH_DIR


class FFNTrainer:
    def __init__(self, model=None):
        self.model = model
        self.model.to(DEVICE)
        # self.model.apply(self.init_weights)

        self.criterion = nn.CrossEntropyLoss()

        if OPTIMIZER.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)

        self.train_loader, self.val_loader, self.test_loader = DataParser().data_loading()

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

        self.save_model()

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

    def predict(self):
        print('Predicting\n')
        if USE_CUDA:
            self.model.cuda()
            long_tensor = torch.cuda.LongTensor
            float_tensor = torch.cuda.FloatTensor
        else:
            long_tensor = torch.LongTensor
            float_tensor = torch.FloatTensor

        test_batch_loss = 0
        test_loss = 0
        pred_labels, loss_list = [], []

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

                    test_batch_loss += t_loss.item()
                    loss_cast = t_loss.tolist()
                    loss_str = str(loss_cast)
                    loss_list.append(loss_str)
                    pbar.set_postfix({'loss': t_loss.item()})
                    pbar.update(x_test.shape[0])
                    y_pred_softmax = torch.log_softmax(pred, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    pred_labels.append(y_pred_tags)
                    # print('Predicted cancer type for patient ', patient[person], 'is: ', pred_labels[person])
                i += 1
            test_loss = test_batch_loss / len(self.val_loader)
            test_loss = round(test_loss, 2) * 100

            self.save_predictions(loss_list)
        print('Prediction successful')

    def save_model(self):
        if not os.path.exists(MODEL_PATH_DIR):
            os.makedirs(MODEL_PATH_DIR)

        torch.save(self.model.state_dict(), MODEL_PATH)

    def save_predictions(self, loss_list):
        save_path = MODEL_PATH_DIR
        if not os.path.exists(MODEL_PATH_DIR):
            os.makedirs(MODEL_PATH_DIR)

        file_name = 'prediction_losses'
        prediction_losses = os.path.join(save_path, file_name + ".txt")

        file1 = open(prediction_losses, "w")

        with open("prediction_losses.txt", "w") as outfile:
            outfile.write("\n".join(str(item) for item in loss_list))
        outfile = open("prediction_losses.txt", "r")
        file1.write(outfile.read())
        file1.close()
        print('Save successful')
