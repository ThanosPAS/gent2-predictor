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
        path_list = []
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
            path_list +=path
            pickle.dump(tensor, open(os.path.join(path, f'{patient}.p'), 'wb'))

    def unpickling (cancer_type):
        cancer_dict = {}
        #Is a path for each subtype(dubbed GROUP_PATH) available somehow?
        for person in GROUP_PATH:
            with open('person', 'rb') as f:
            cancer_dict.update(pickle.load(f))
            #Idk how to make the patient numbers the keys of the dict - if this operation doesn't do it already

        return cancer_dict



def data_loading():
    #How should we distribute the cancer_dict values into in the below variables?
    #For a list of keys assign for example to x_train these values?

    #Training data
    #for path in path_list:
    x_train = DataLoader(cancer_dict[expression values], )
    y_train = DataLoader(cancer_dict[some patient IDs], )
    #Validating the models performance
    x_val = DataLoader(cancer_dict[expression values], )
    y_val = DataLoader(cancer_dict[some patient IDs], )
    #Never seen data by the model
    x_test = DataLoader(cancer_dict[expression values], )
    y_test = DataLoader(cancer_dict[some patient IDs], )

    #In case the tensors need flattening
    #x_train = x_train.reshape(x_train.shape[0], -1)
    #x_val = x_val.reshape(x_val.shape[0], -1)
    #x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = data_loading()
