from DiabetsIndian import GetData
import numpy as np


def prediction_model(model, count):
    if count > 767:
        return 'max of count is 767'
    dateset = GetData.load_data()
    array = np.array(dateset)
    x = array[:, 0:8]
    y = array[:, 8:]
    sum = 0
    predictions = model.predict_classes(x)
    for i in range(count):
        if predictions[i] == y[i]:
            sum += 1
    return str(int((sum * 100) / count)) + '%'
