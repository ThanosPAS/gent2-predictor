import pickle

import pandas as pd
from gent2_predictor.settings import STRUCTURE_PATH, DATA_PATH


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
        pickle.dump(data, open('data.p', 'wb'))
