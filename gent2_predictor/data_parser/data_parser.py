import pickle
import os
import torch
import errno

import pandas as pd
from gent2_predictor.settings import STRUCTURE_PATH, DATA_PATH, DATA_DIR


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def _parse_structure():
        structure = pd.read_excel(STRUCTURE_PATH)

        structure.columns = ['index', 'cancer_type', 'patient']
        structure = structure.drop(['index'], axis=1)
        structure['cancer_type'] = structure['cancer_type'].ffill()

        return structure

    def parse_cancer_files(self):
        data = pd.read_csv(DATA_PATH, sep=',')
        structure = self._parse_structure()

        data = data.merge(structure, on='patient')

        return data

    @staticmethod
    def pickle_data(data):
        for patient in data['patient'].unique():
            patient_data = data.loc[data['patient'] == patient]
            cancer_type = patient_data['cancer_type'].unique()[0]
            patient_data = patient_data['expression'].values
            tensor = torch.from_numpy(patient_data)
            # target = torch.zeros(149 * 149)
            # target[:21920] = tensor
            # target = torch.reshape(target, (149, 149))
            # target = normalize(target)

            path = os.path.join(DATA_DIR, cancer_type)

            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            pickle.dump(tensor, open(os.path.join(path, f'{patient}.p'), 'wb'))
