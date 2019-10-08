import pandas as pd
import numpy as np


class __array_start_counter__():
    counter = 0


def load_data():
    dataset = pd.read_csv('/home/hafez/Documents/AI/DataSets/pima-indians-diabetes.data.csv')
    return dataset


def get_percent_data(x=100):
    counter = int((767 * x) / 100) + 1
    dataset = load_data()
    array = np.array(dataset)
    if __array_start_counter__.counter + counter >= 767:
        return 'out of range'
    x_train = array[__array_start_counter__.counter:(__array_start_counter__.counter + counter), 0:8]
    y_train = array[__array_start_counter__.counter:(__array_start_counter__.counter + counter), 8:]
    __array_start_counter__.counter += counter
    x_test = array[__array_start_counter__.counter:, 0: 8]
    y_test = array[__array_start_counter__.counter:, 8:]
    return x_train, y_train, x_test, y_test
