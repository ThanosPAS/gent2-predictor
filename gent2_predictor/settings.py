import os

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')

CANCER_DATA_DIR = os.path.join(DATA_DIR, 'cancer_data')
FULL_DATA_DIR = os.path.join(DATA_DIR, 'cancer_data', 'full')
LANDMARKS_DATA_DIR = os.path.join(DATA_DIR, 'cancer_data', 'landmarks')

USE_FULL_DATA = False

MODEL_PATH_DIR = os.path.join(DATA_DIR, 'models')

PLOTS_PATH_DIR = os.path.join(DATA_DIR, 'plots')
PREDICTIONS_PATH_DIR = os.path.join(DATA_DIR, 'predictions')

if not os.path.exists(MODEL_PATH_DIR):
    os.makedirs(MODEL_PATH_DIR)

if not os.path.exists(PLOTS_PATH_DIR):
    os.makedirs(PLOTS_PATH_DIR)

if not os.path.exists(PREDICTIONS_PATH_DIR):
    os.makedirs(PREDICTIONS_PATH_DIR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

EPOCHS = 3
LEARNING_RATE = 0.0001

# OPTIMIZER = 'ADAM'
OPTIMIZER = 'SGD'

INIT_METHOD = 'xavier_uniform_'
# INIT_METHOD = 'xavier_normal_'
# INIT_METHOD = 'kaiming_uniform_'
# INIT_METHOD = 'kaiming_normal_'
# INIT_METHOD = 'orthogonal_'
# INIT_METHOD = 'sparse_'

L2_REG = 0
MOMENTUM = 0

TARGET_LABELS = {
    'liver'   : 0,
    'colon'   : 1,
    'lung'    : 2,
    'breast'  : 3,
    'stomach' : 4,
    'leukemia': 5,
    'lymphoma': 6,
    'normal'  : 7,
}

MODEL_SELECTOR = 'FULL_FFN'


# MODEL_SELECTOR = 'BASELINE_FFN'


def create_pathname():
    model_name = input(
        "Enter the model you want to predict on (e.g.: 'ffn_2020-12-08 12:18:22.pth'):")
    model_path = os.path.join(
        MODEL_PATH_DIR,
        model_name
    )
    return model_name, model_path


LANDMARK_URL = 'https://api.clue.io/api/genes?filter={%22where%22:{%22l1000_type%22:%22landmark%22}}&user_key=18c46f38bfd42d8c229d1866dced8a89'
