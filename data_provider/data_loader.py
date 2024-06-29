'''
"Dynamic Feature-Selection" Code base 

Code author: Hyeryn Park (qkrgpfls1201@gmail.com), Hyuntae Kim (soodaman97@cau.ac.kr)
----------------------------------------------------------

data_loader.py

(1) Data preprocessing function
    - moving_avg 
    - UserMinMaxScaler
    - UserStandardScaler
    
(2) Pytorch Dataset 
    - Dataset_ETRI
'''

import os
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from tqdm import tqdm 
warnings.filterwarnings('ignore')


#-- ETRI 
# data processing moving average 
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

# custom scaler
class UserMinMaxScaler:
    def __init__(self):
        self.max_num = -np.inf
        self.min_num = np.inf

    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.max_num = np.max(arr)
        self.min_num = np.min(arr)

    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.max_num = np.max(arr)
        self.min_num = np.min(arr)

        return (arr - self.min_num) / (self.max_num - self.min_num)

    def transform(self, arr):
        return (arr - self.min_num) / (self.max_num - self.min_num)
    
    def inverse_transform(self, arr): 
        return arr * (self.max_num - self.min_num) + self.min_num

class UserStandardScaler:
    def __init__(self):
        self.mean_num = None
        self.std_num = None

    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.mean_num = np.mean(arr)
        self.std_num = np.std(arr)

    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.mean_num = np.mean(arr)
        self.std_num = np.std(arr)

        return (arr - self.mean_num) / self.std_num

    def transform(self, arr):
        return (arr - self.mean_num) / self.std_num
    
    def inverse_transform(self, arr):
        return (arr * self.std_num) + self.mean_num

class Dataset_ETRI(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 in_w=7, out_w=3, scale=True, 
                 scale_type='minmax', tanh=0,):
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]    
        self.scale_type = scale_type    
        self.tanh = tanh
        self.label_len = size 
        self.scale = scale
        self.root_path = root_path

        self.in_w = in_w
        self.out_w = out_w

        self.__read_data__()       

    def __read_data__(self):
        if self.scale_type == 'minmax':
            self.scaler = UserMinMaxScaler()
        elif self.scale_type == 'standard':
            self.scaler = UserStandardScaler()

        npz  = np.load(f'{self.root_path}/BM-0006.zip')['RSRPTx32R8T10']
        data = np.asarray(npz)
        del npz

        data = data.reshape(100000,10,32,-1)

        tr_data, te_data = train_test_split(data, test_size=0.2)
        tr_data, va_data = train_test_split(tr_data, test_size=0.2)

        # scaler
        self.scaler.fit(tr_data)
        tr_data = self.scaler.transform(tr_data)
        te_data = self.scaler.transform(te_data)
        va_data = self.scaler.transform(va_data)

        if self.set_type == 0: # train 
            self.data = tr_data
        elif self.set_type == 1: # val 
            self.data = va_data
        elif self.set_type == 2: # test 
            self.data = te_data
                        

    def __getitem__(self, index):

        if self.tanh == 1:
            seq = self.data[index] * 2 - 1
        else: 
            seq = self.data[index]

        return seq
    
    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)