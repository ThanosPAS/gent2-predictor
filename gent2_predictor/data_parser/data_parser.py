import errno
import os
import pickle

import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, random_split

from gent2_predictor.data_parser.gent2_dataset import Gent2Dataset
from gent2_predictor.settings import PROCESSED_DATA_DIR, \
    FULL_DATA_DIR, LANDMARKS_DATA_DIR, RAW_DATA_DIR, CANCER_DATA_DIR


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_structures():
        structure = pd.read_excel(os.path.join(RAW_DATA_DIR, 'structure.xlsx'))

        structure.columns = ['index', 'cancer_type', 'patient']
        structure = structure.drop(['index'], axis=1)
        structure['cancer_type'] = structure['cancer_type'].ffill()
        structure['cancer_type'] = structure['cancer_type'].str.lower()

        more_structure = pd.read_excel(os.path.join(RAW_DATA_DIR, 'more_structure.xlsx'))
        more_structure = more_structure.drop(['number'], axis=1)
        more_structure = more_structure.rename({'disease': 'cancer_type'}, axis=1)

        structure = pd.concat([structure, more_structure])

        if not os.path.exists(PROCESSED_DATA_DIR):
            try:
                os.makedirs(PROCESSED_DATA_DIR)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        structure.to_csv(os.path.join(PROCESSED_DATA_DIR, 'full_structure.csv'), index=False)

    @staticmethod
    def parse_data():
        data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'data.csv'))
        more_data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'more_data.csv'), header=None)

        cols = ['probe_id', 'patient', 'expression']
        data.columns = cols
        more_data.columns = cols
        data = pd.concat([data, more_data])
        data = data.drop_duplicates(subset=cols)

        return data

    def pickle_data(self):
        try:
            os.makedirs(os.path.join(CANCER_DATA_DIR, 'full'))
            os.makedirs(os.path.join(CANCER_DATA_DIR, 'landmarks'))
        except Exception as ex:
            print(str(ex))

        data = self.parse_data()
        landmarks = self.get_landmarks()
        landmarks_data = data.merge(landmarks, on='probe_id')

        for patient in data['patient'].unique():
            for dataset, location in [(data, 'full'), (landmarks_data, 'landmarks')]:
                patient_data = dataset.loc[dataset['patient'] == patient]
                patient_data = patient_data['expression'].values
                tensor = torch.from_numpy(patient_data)
                pickle.dump(tensor, open(os.path.join(
                    CANCER_DATA_DIR, location, f'{patient}.p'), 'wb'))

    @staticmethod
    def data_loading(full_data=False):
        dataset = Gent2Dataset(
            os.path.join(PROCESSED_DATA_DIR, 'structure.csv'),
            FULL_DATA_DIR if full_data else LANDMARKS_DATA_DIR
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

    @staticmethod
    def get_landmarks():
        response = requests.get(
            'https://api.clue.io/api/genes?filter={%22where%22:{%22l1000_type%22:%22landmark%22}}&user_key=18c46f38bfd42d8c229d1866dced8a89')

        df = pd.DataFrame(response.json())
        df = df.rename({'gene_symbol': 'probe_id'}, axis=1)
        df = df['probe_id']

        return df
