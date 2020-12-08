import os
from datetime import datetime

import torch

from gent2_predictor.settings import MODEL_PATH_DIR, MODEL_FILENAME


class Trainer:
    def __init__(self):
        pass

    def save_model(self, model, model_type):
        if not os.path.exists(MODEL_PATH_DIR):
            os.makedirs(MODEL_PATH_DIR)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_path = f'{model_type}_{timestamp}.pth'
        model_path = os.path.join(MODEL_PATH_DIR, model_path)

        torch.save(model.state_dict(), model_path)

    def save_predictions(self, loss_list):
        if not os.path.exists(MODEL_PATH_DIR):
            os.makedirs(MODEL_PATH_DIR)

        file_name = f'prediction_losses_{MODEL_FILENAME}.txt'
        file = os.path.join(MODEL_PATH_DIR, file_name)

        with open(file, "w") as outfile:
            outfile.write("\n".join(str(item) for item in loss_list))

        print('Prediction losses saved successfully')
