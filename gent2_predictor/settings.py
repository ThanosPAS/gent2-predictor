import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATA_PATH = f'{DATA_DIR}/data.csv'
STRUCTURE_PATH = f'{DATA_DIR}/Dataset structure.xlsx'
