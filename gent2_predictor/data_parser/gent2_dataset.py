import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset

from gent2_predictor.settings import DEVICE


class Gent2Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Structure of one element:
        {'data': tensor([ 85,  19, 482,  ..., 281, 122, 465]), 'cancer_type': 'LIVER'}

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cancer_dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cancer_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_file = os.path.join(self.root_dir, f'{self.cancer_dataframe.iloc[idx, 1]}.p')
        cancer_type = self.cancer_dataframe.iloc[idx, 0]

        with open(patient_file, 'rb') as f:
            data = pickle.load(f)

        sample = {'data': data.to(DEVICE), 'cancer_type': cancer_type}

        if self.transform:
            sample = self.transform(sample)

        return sample
