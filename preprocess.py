import orjson
import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class DatasetProcess:
    def __init__(self):
        self.file_list = os.listdir('data')
        self.save_path = 'data.json'
        self.data_path = 'data/'
        self.values = None
        self.train_data = None
        self.test_data = None
        self.dataset()

    def load(self, path):
        print(self.data_path + path)
        return pd.read_excel(self.data_path + path)

    def dataset(self):
        data = None
        for file in self.file_list:
            contents = self.load(file)
            self.values = contents.keys() if self.values is None else self.values
            chla = np.array(contents['Chla'])
            temp_data = contents.values
            data_index = np.argwhere(chla != '--')
            useful_data = temp_data[data_index.reshape(-1), 2:]
            data = useful_data if data is None else np.concatenate((data, useful_data), axis=0)
        data[np.argwhere(data[:, 1] < 0), 1] += 360
        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=42)

    def save(self):
        data = {'value': self.values.values.tolist(), 'train': self.train_data.tolist(),
                'test': self.test_data.tolist()}
        with open(self.data_path + self.save_path, 'w') as f:
            f.write(orjson.dumps(data).decode())
        f.close()
        print('----Save Finish----')

class ChlDataset(Dataset):
    def __init__(self, train):
        with open('data/data.json', 'r') as f:
            data = orjson.loads(f.read())
            train_data = np.array(data['train']).astype(float)
            self.data = train_data if train else np.array(data['test']).astype(float)
        self.feature_index = [0, 1, 2, -2, -1]
        self.mean = np.mean(train_data[:, 3:-2], axis=0)
        self.std = np.std(train_data[:, 3:-2], axis=0)
        self.data[:, 3:-2] = (self.data[:, 3:-2] - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, self.feature_index]
        y = np.concatenate((self.data[index, 3:303], self.data[index, -5:-2]))
        return torch.from_numpy(x), torch.from_numpy(y)
