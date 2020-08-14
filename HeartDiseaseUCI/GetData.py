import pandas as pd
import numpy as np


class __array_fisrt_index__():
    counter = 0


def load_data():
    return pd.read_csv('heart.csv')


def get_data(percent):
    dataset = load_data()
    array = np.array(dataset)
    array_end_index = int(303 * percent / 100) + 1
    train_x = array[__array_fisrt_index__.counter:(__array_fisrt_index__.counter + array_end_index), 0:13]
    train_y = array[__array_fisrt_index__.counter:(__array_fisrt_index__.counter + array_end_index), 13:]
    test_x = array[(__array_fisrt_index__.counter + array_end_index):, 0:13]
    test_y = array[(__array_fisrt_index__.counter + array_end_index):, 13:]
    return train_x, train_y, test_x, test_y
