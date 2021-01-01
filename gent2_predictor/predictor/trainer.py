import os
from datetime import datetime
import csv
import pandas as pd
import torch
from sklearn.metrics import classification_report
from gent2_predictor.settings import MODEL_PATH_DIR, PREDICTIONS_PATH_DIR


class Trainer:
    def __init__(self):
        pass


    def save_model(self, model, model_type):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        model_name = f'{model_type}_{timestamp}.pth'
        model_path = os.path.join(MODEL_PATH_DIR, model_name)

        torch.save(model.state_dict(), model_path)

        return model_name

    def save_predictions(self, filename, loss_list,train_loss=None, valid_loss=None, y_test_arr=None, pred_arr=None,mode=True):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        if mode:
            self.model_name=filename
            file_name = f'prediction_losses_{self.model_name}.txt'
            file = os.path.join(PREDICTIONS_PATH_DIR, file_name)
            with open(file, "w") as outfile:
                outfile.write("\n".join(str(item) for item in loss_list))
            print('Prediction losses saved successfully')

            report_name = f'classification_report_{self.model_name}.csv'
            file = os.path.join(PREDICTIONS_PATH_DIR, report_name)
            report = classification_report(y_test_arr, pred_arr, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(file)
        else:
            self.model_name = filename
            file_name = f'train-val_losses_{self.model_name}_{timestamp}.csv'
            file = os.path.join(PREDICTIONS_PATH_DIR, file_name)
            with open(file, "w", newline='') as outfile:
                fieldnames = ['train_loss', 'validation_loss']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for line in range(len(train_loss)):
                    writer.writerow({'train_loss': train_loss[line], 'validation_loss': valid_loss[line]})
                #outfile.write("\n".join(str(item) for item in loss_list))
            print('Train & validation losses saved successfully')
