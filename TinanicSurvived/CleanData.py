import numpy as np


def clean_data(dataset):
    array_dataset = np.array(dataset)
    array_dataset = np.where(array_dataset == 'male', 1, array_dataset)
    array_dataset = np.where(array_dataset == 'female', 0, array_dataset)
    return array_dataset


