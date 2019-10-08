import numpy as np
from ChurnModeling import GetData as gd

"""
concatenate in numpy
"""
# a = np.array([[0], [5]])
# b = np.array([[0, 2, 4], [6, 8, 10]])
# c = np.concatenate((a, b), 1)
# d = np.array([1, 2, 3, 4])
# print(d.shape)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(c)


datasets = gd.load_data()
datasets = datasets[['Gender', 'Age']]
array = np.array(datasets)
list = []
for i in range(array.__len__()):
    if array[i, i + 1, 0:1] == 'Female':
        # list.append(array[i, i + 1, 0:1])
        print(0)
    else:
        # list.append(array[i, i + 1, 0:1])
        print(1)
print(list)
