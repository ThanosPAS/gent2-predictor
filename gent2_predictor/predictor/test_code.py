from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.criterion = nn.BCELoss(size_average=True)
        self.optimizer = torch.optim.SGD(self.log_reg.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, momentum=MOMENTUM)
        self.train_loader, self.val_loader, self.test_loader = DataParser().data_loading()
        self.log_reg = LogisticRegression(C=1.0, random_state=0, fit_intercept=True, multi_class='multinomial',
                                     penalty='none',
                                     n_jobs=-1, solver='saga').fit(x_train, y_train)
    def log_training(self):

        for epoch in range(EPOCHS):
            #Data loading
            for train_person, val_person in self.train_loader, self.val_loader :
                x_train_numpy = train_person['data']
                x_train = np.array(x_train_numpy, dtype=np.int16)
                x_train = x_train.reshape(-1, 1)

                x_val_numpy = val_person['data']
                x_val = np.array(x_val_numpy, dtype=np.int16)
                x_val = x_val.reshape(-1, 1)

                y_values = np.array([i for i in range(6)])
                y_train = np.array(y_values, dtype=np.int8)
                y_train = y_train.reshape(-1, 1)
                x_train = Normalizer().fit_transform(x_train)


                #Training starts...
                #Make sure x_val, y_val are correct
                self.log_reg.train()
                log_pred = self.log_reg.predict(x_val)
                loss = self.criterion(log_pred, self.y_val)
                '''
                if torch.cuda.is_available():
                    x_train = Variable(torch.from_numpy(x_train).cuda())
                    y_train = Variable(torch.from_numpy(y_train).cuda())
                else:
                    x_train = Variable(torch.from_numpy(x_train))
                    y_train = Variable(torch.from_numpy(y_train))
                    '''
                loss.backward()
                self.optimizer.step()


log_reg = LogisticRegression()


confusion_matrix(self.y_test, log_pred)




#{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None,
 #'max_iter': 100, 'multi_class': 'multinomial', 'n_jobs': None, 'penalty': 'none', 'random_state': 0,
 #'solver': 'newton-cg', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}