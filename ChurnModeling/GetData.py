import pandas as pd
import numpy as np


def load_data():
    dataset = pd.read_csv('Churn_Modelling.csv')
    return dataset


def get_data(features, percent,index_feature):
    dataset = load_data()
    dataset1 = dataset[features]
    y = int((10000 * percent) / 100) + 1
    first_index = 0
    array = np.array(dataset1)
    train_x = array[first_index:y, 0:index_feature]
    train_y = array[first_index:y, index_feature:]
    first_index += y
    test_x = array[first_index:, 0:index_feature]
    test_y = array[first_index:, index_feature:]
    return train_x, train_y, test_x, test_y


def clean_data():
    ...
    # array_gender = np.array([[]])
    # for i in range(0, array.__len__()):
    #     e = array[i:i + 1, 0:1]
    #     if e == [['Female']]:
    #         array_gender = np.append(array_gender, 0)
    #     else:
    #         array_gender = np.append(array_gender, 1)
    #
    # print(array_gender.shape)
    # clean_array = np.concatenate((array_gender,array),1)
    # print(clean_array)
