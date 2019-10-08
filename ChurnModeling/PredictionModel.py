from ChurnModeling import GetData
import numpy as np


def prediction_model(model, count, features, index_feature):
    if count > 10000:
        return 'max of count is 10000'
    dataset = GetData.load_data()
    dataset1 = dataset[features]
    array = np.array(dataset1)
    x = array[:, 0:index_feature]
    y = array[:, index_feature:]
    predictions = model.predict_classes(x)
    sum = 0
    for i in range(count):
        if predictions[i] == y[i]:
            sum += 1
    return str(int((sum * 100) / count)) + '%'
