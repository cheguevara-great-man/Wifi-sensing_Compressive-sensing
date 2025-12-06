import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import glob, os
from scipy.signal import resample
from scipy.interpolate import interp1d, Rbf
from scipy import interpolate

def upsample_signal(data, target_length, method='linear'):
    """
    对二维或三维信号进行上采样
    支持方法：linear, cubic, nearest, idw, rbf
    data.shape = (N, F) 或 (N, 250, 90)
    """
    if data.ndim == 2:
        T, F = data.shape
        x_old = np.arange(T)
        x_new = np.linspace(0, T-1, target_length)
        data_upsampled = np.zeros((target_length, F))
        for j in range(F):
            y = data[:, j]
            if method in ['linear', 'cubic', 'nearest']:
                f = interpolate.interp1d(x_old, y, kind=method, fill_value="extrapolate")
                data_upsampled[:, j] = f(x_new)
            elif method == 'idw':
                y_new = []
                for xq in x_new:
                    dist = np.abs(x_old - xq)
                    dist[dist == 0] = 1e-8
                    weights = 1 / dist
                    y_pred = np.sum(weights * y) / np.sum(weights)
                    y_new.append(y_pred)
                data_upsampled[:, j] = y_new
            elif method == 'rbf':
                rbf_func = interpolate.Rbf(x_old, y, function='multiquadric')
                data_upsampled[:, j] = rbf_func(x_new)
        return data_upsampled
    elif data.ndim == 3:
        # (N, 250, 90)
        N, T, F = data.shape
        x_old = np.arange(N)
        x_new = np.linspace(0, N-1, target_length)
        data_upsampled = np.zeros((target_length, T, F))
        for t in range(T):
            for f in range(F):
                y = data[:, t, f]
                if method in ['linear', 'cubic', 'nearest']:
                    f_interp = interpolate.interp1d(x_old, y, kind=method, fill_value="extrapolate")
                    data_upsampled[:, t, f] = f_interp(x_new)
                elif method == 'idw':
                    y_new = []
                    for xq in x_new:
                        dist = np.abs(x_old - xq)
                        dist[dist == 0] = 1e-8
                        weights = 1 / dist
                        y_pred = np.sum(weights * y) / np.sum(weights)
                        y_new.append(y_pred)
                    data_upsampled[:, t, f] = y_new
                elif method == 'rbf':
                    rbf_func = interpolate.Rbf(x_old, y, function='multiquadric')
                    data_upsampled[:, t, f] = rbf_func(x_new)
        return data_upsampled
    else:
        raise ValueError("data维度错误，必须是2D或3D")


downsample_rate = 5
method = 'nearest'
def UT_HAR_dataset(root_dir):

    data_list = glob.glob(root_dir+'UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            # 降采样
            data = data[::downsample_rate, :]
            '''
            target_len = len(data)
            #降采样
            data = data[::downsample_rate, :]
            #上采样
            data_up = upsample_signal(data, target_len, method=method)
            data = data_up
            '''


            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
            # 降采样
            label = label[::downsample_rate]
            '''
            target_label_len = len(label)
            # 降采样
            label = label[::downsample_rate]
            #上采样
            label = upsample_signal(label.reshape(-1, 1), target_label_len, method='nearest')
            WiFi_data[label_name] = torch.Tensor(label.squeeze())
            '''
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y

