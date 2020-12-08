import os

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')

DATA_PATH = os.path.join(RAW_DATA_DIR, 'data.csv')
STRUCTURE_PATH = os.path.join(RAW_DATA_DIR, 'Dataset structure.xlsx')

CANCER_DATA_DIR = os.path.join(DATA_DIR, 'cancer_data')

MODEL_PATH_DIR = os.path.join(DATA_DIR, 'models')
MODEL_FILENAME = 'ffn_2020-12-08 12:18:22.pth'  # e.g.: 'ffn_2020-12-08 12:18:22.pth'

MODEL_PATH = os.path.join(
    MODEL_PATH_DIR,
    MODEL_FILENAME
)

PLOTS_PATH_DIR = os.path.join(DATA_DIR, 'plots')

if not os.path.exists(MODEL_PATH_DIR):
    os.makedirs(MODEL_PATH_DIR)

if not os.path.exists(PLOTS_PATH_DIR):
    os.makedirs(PLOTS_PATH_DIR)

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
    'LIVER'  : 0,
    'COLON'  : 1,
    'LUNG'   : 2,
    'BREAST' : 3,
    'STOMACH': 4,
    'NORMAL' : 5
}
