import os
from datetime import datetime

import torch

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

    def save_predictions(self, loss_list):
        model_name=input('Predictions filename(preferably same as model name):')
        file_name = f'prediction_losses_{model_name}.txt'
        file = os.path.join(PREDICTIONS_PATH_DIR, file_name)


        with open(file, "w") as outfile:
            outfile.write("\n".join(str(item) for item in loss_list))

        print('Prediction losses saved successfully')
