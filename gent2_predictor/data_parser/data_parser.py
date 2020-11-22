<<<<<<< Updated upstream
=======
from sklearn.preprocessing import normalize as norm
import pickle
import os
import torch
>>>>>>> Stashed changes
import errno
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from gent2_predictor.data_parser.gent2_dataset import Gent2Dataset
from gent2_predictor.settings import STRUCTURE_PATH, DATA_PATH, PROCESSED_DATA_DIR, CANCER_DATA_DIR


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_structure():
        structure = pd.read_excel(STRUCTURE_PATH)

        structure.columns = ['index', 'cancer_type', 'patient']
        structure = structure.drop(['index'], axis=1)
        structure['cancer_type'] = structure['cancer_type'].ffill()
        if not os.path.exists(PROCESSED_DATA_DIR):
            try:
                os.makedirs(PROCESSED_DATA_DIR)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        structure.to_csv(os.path.join(PROCESSED_DATA_DIR, 'structure.csv'), index=False)

    @staticmethod
    def pickle_data():
        data = pd.read_csv(DATA_PATH)

        if not os.path.exists(CANCER_DATA_DIR):
            try:
                os.makedirs(CANCER_DATA_DIR)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        for patient in data['patient'].unique():
            patient_data = data.loc[data['patient'] == patient]
            patient_data = patient_data['expression'].values
            tensor = torch.from_numpy(patient_data)

            pickle.dump(tensor, open(os.path.join(CANCER_DATA_DIR, f'{patient}.p'), 'wb'))

    @staticmethod
    def data_loading():
        dataset = Gent2Dataset(
            os.path.join(PROCESSED_DATA_DIR, 'structure.csv'),
            CANCER_DATA_DIR
        )

        dataset_length = len(dataset)
        val_size = round(dataset_length * 0.15)
        test_size = round(dataset_length * 0.15)
        train_size = dataset_length - val_size - test_size

        train_set, val_size, test_size = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_set)
        val_loader = DataLoader(val_size)
        test_loader = DataLoader(test_size)

        return train_loader, val_loader, test_loader
