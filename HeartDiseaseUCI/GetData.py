import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class __array_fisrt_index__():
    counter = 0


def load_data():
    return pd.read_csv('heart.csv')


def get_data(percent):
    df = load_data()
    y = df['target']
    df.drop(columns='target', inplace=True)
    x = df
    return train_test_split(x, y, random_state=0, train_size=percent / 100)
