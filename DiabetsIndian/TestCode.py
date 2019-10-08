import numpy as np
import pandas as pd

dataset = pd.read_csv('/home/hafez/Downloads/pima-indians-diabetes.data.csv')
array = np.array(dataset[0:10])
array1 = np.array([1, 2, 3, 4, 5, 5])
print(array[1:8, 0:9])


class counter():
    counter = 0




print(counter.counter)
x = 50
counter.counter += x
print(counter.counter)
