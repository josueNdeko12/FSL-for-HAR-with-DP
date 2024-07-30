import urllib.request
import zipfile
from numpy import std, dstack
from pandas import read_csv
import torch
import os

dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
dataset_dir = 'HARDataset'
dataset_zip = 'UCI_HAR_Dataset.zip'


# Function to download and extract the dataset
def download_and_extract_dataset(url, dest_path, zip_path):
    if not os.path.exists(dest_path):
        print('Downloading dataset...')
        urllib.request.urlretrieve(url, zip_path)
        print('Extracting dataset...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print('Dataset downloaded and extracted.')
    else:
        print('Dataset already exists.')


# Function to load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# Function to load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    loaded = dstack(loaded)
    return loaded


# Function to load a dataset group (train or test)
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    filenames = [
        'total_acc_x_', 'total_acc_y_', 'total_acc_z_',
        'body_acc_x_', 'body_acc_y_', 'body_acc_z_',
        'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_'
    ]
    filenames = [filepath + name + group + '.txt' for name in filenames]
    x = load_group(filenames)
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return x, y


# Function to load the entire dataset
def load_dataset(prefix=''):
    train_x, train_y = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
    test_x, test_y = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    train_y = torch.eye(6)[train_y.squeeze() - 1]
    test_y = torch.eye(6)[test_y.squeeze() - 1]
    return train_x, train_y, test_x, test_y