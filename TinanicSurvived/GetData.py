import pandas as pd
import numpy as np
from TinanicSurvived import CleanData


def load_data():
    return pd.read_csv('/home/hafez/Documents/AI/DataSets/TitinicSurvived.csv')


def get_data(features, percent=100, predict=False):
    len_feature = (len(features) - 1)
    datasets = load_data()[features]
    datasets_array = CleanData.clean_data(datasets)
    if predict == True:
        return datasets_array
    index_end_train = int((percent * 891) / 100) + 1
    train_x = datasets_array[0:index_end_train, 0:(len_feature)]
    train_y = datasets_array[0:index_end_train, len_feature:]
    test_x = datasets_array[index_end_train:, 0:len_feature]
    test_y = datasets_array[index_end_train:, len_feature:]
    return train_x, train_y, test_x, test_y
