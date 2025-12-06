import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import glob, os
from scipy.signal import resample
from scipy.interpolate import interp1d, Rbf
from scipy import interpolate
from scipy.special import gammaln

def upsample_signal(data, target_length, method='linear'):
    """
    对二维或三维信号进行上采样
    支持方法：linear, cubic, nearest, idw, rbf
    data.shape = (N, F) 或 (N, 250, 90)
    """

    def fast_idw(x_old, y, x_new, k=10):
        result = np.zeros_like(x_new)
        for i, xq in enumerate(x_new):
            idx = np.argsort(np.abs(x_old - xq))[:k]
            dist = np.abs(x_old[idx] - xq)
            dist[dist == 0] = 1e-8
            w = 1 / dist
            w /= np.sum(w)
            result[i] = np.sum(w * y[idx])
        return result

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
                data_upsampled[:, t, f] = fast_idw(x_old, y, x_new, k=10)
            elif method == 'rbf':
                cs = interpolate.CubicSpline(x_old, y)
                data_upsampled[:, t, f] = cs(x_new)
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

                    dist_matrix = np.abs(x_new[:, None] - x_old[None, :])  # [target_len, N]
                    dist_matrix[dist_matrix == 0] = 1e-8
                    weights = 1 / dist_matrix
                    weights /= np.sum(weights, axis=1, keepdims=True)
                    data_upsampled[:, t, f] = np.dot(weights, y)
                elif method == 'rbf':
                    rbf_func = interpolate.Rbf(x_old, y, function='multiquadric', smooth=0.1)
                    data_upsampled[:, t, f] = rbf_func(x_new)

                elif method=='spline':
                    spline = interpolate.InterpolatedUnivariateSpline(x_old, y)
                    data_upsampled[:, t, f] = spline(x_new)
                elif method=='akima':
                    akima = interpolate.Akima1DInterpolator(x_old, y)
                    data_upsampled[:, t, f] = akima(x_new)
        return data_upsampled
    else:
        raise ValueError("data维度错误，必须是2D或3D")

downsample_rate = 2
method = 'spline'
def UT_HAR_dataset_down_Up(root_dir):

    data_list = glob.glob(root_dir+'UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            target_len = len(data)
            # 降采样
            data = data[::downsample_rate, :]
            #上采样
            data_up = upsample_signal(data, target_len, method=method)
            data = data_up

            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
            target_label_len = len(label)
            # 降采样
            label = label[::downsample_rate]
            #上采样
            label = upsample_signal(label.reshape(-1, 1), target_label_len, method='nearest')
        WiFi_data[label_name] = torch.Tensor(label.squeeze())

        #WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data

downsample_rate = 5
def UT_HAR_dataset_down(root_dir):

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


def random_sample_signal(data, sample_ratio=0.5, method='uniform', mu=None, sigma=None, lam=None):

    N = data.shape[0]
    num_samples = int(N * sample_ratio)

    if num_samples <= 0:
        raise ValueError("采样比例过小，请增大 sample_ratio。")

    # 高斯分布采样
    elif method == 'gaussian':
        if mu is None:
            mu = N / 2
        if sigma is None:
            sigma = N / 6
        sampled_idx = np.random.normal(mu, sigma, num_samples).astype(int)
        sampled_idx = np.clip(sampled_idx, 0, N - 1)
        sampled_idx = np.unique(sampled_idx)
        while len(sampled_idx) < num_samples:
            extra = np.random.normal(mu, sigma, num_samples - len(sampled_idx)).astype(int)
            extra = np.clip(extra, 0, N - 1)
            sampled_idx = np.unique(np.concatenate([sampled_idx, extra]))
        sampled_idx = np.sort(sampled_idx[:num_samples])

    # 泊松分布采样
    elif method == 'poisson':
        ratio = 0.5
        num_samples = int(np.floor(N * ratio))
        if lam is None:
            lam = max(1.0, N / 4.0) # 控制采样集中区

        x = np.arange(N)  # 支持索引 0..N-1

        # 对数域计算泊松 pmf： log P(x) = x*log(lam) - lam - log(x!)
        log_p = x * np.log(lam) - lam - gammaln(x + 1)

        # 为稳定性，减去最大值再 exp（避免溢出）
        log_p = log_p - np.max(log_p)
        p = np.exp(log_p)

        # 归一化概率
        p = p / np.sum(p)

        # 根据概率一次性抽样（不放回）
        sampled_idx = np.random.choice(N, size=num_samples, replace=False, p=p)
        sampled_idx = np.sort(sampled_idx)



    else:
        raise ValueError(f"未知的采样方式: {method}")

    return data[sampled_idx], sampled_idx


def UT_HAR_dataset_Randomdown_Up(root_dir):

    data_list = glob.glob(root_dir+'UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            target_len = len(data)
            # 降采样
            data_sampled, sampled_idx = random_sample_signal(data, sample_ratio=0.5, method='poisson')
            data = data_sampled
            #上采样
            data_up = upsample_signal(data, target_len, method='spline')
            data = data_up

            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
            target_label_len = len(label)
            # 降采样
            label_sampled, idx = random_sample_signal(label, sample_ratio=0.5, method='poisson')
            label = label_sampled
            #上采样
            label = upsample_signal(label.reshape(-1, 1), target_label_len, method='nearest')
        WiFi_data[label_name] = torch.Tensor(label.squeeze())

        #WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data



def UT_HAR_dataset_Randomdown(root_dir):

    data_list = glob.glob(root_dir+'UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)

            target_len = len(data)

            data_sampled, sampled_idx = random_sample_signal(data, sample_ratio=0.5, method='poisson')
            x_old = np.linspace(0, 1, len(data_sampled))
            x_new = np.linspace(0, 1, target_len)
            f = interpolate.interp1d(x_old, data_sampled, axis=0, kind='linear', fill_value="extrapolate")
            data_upsampled = f(x_new)

            data = data_upsampled
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)

            label_sampled, idx = random_sample_signal(label, sample_ratio=0.5, method='poisson')
            x_old = np.linspace(0, 1, len(label_sampled))
            x_new = np.linspace(0, 1, len(label))
            f = interpolate.interp1d(x_old, label_sampled, kind='nearest', fill_value="extrapolate")
            label_up = f(x_new)
            label = label_up

        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data







def UT_HAR_dataset(root_dir):

    data_list = glob.glob(root_dir+'UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)

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

