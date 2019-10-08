from HeartDiseaseUCI import GetData
import numpy as np


def prediction_model(model, count):
    if count > 303:
        return 'max of count is 303 '
    dataset = GetData.load_data()
    array = np.array(dataset)
    x = array[:, 0:13]
    y = array[:, 13:]
    sum = 0
    predictions = model.predict_classes(x)
    for i in range(count):
        if predictions[i] == y[i]:
            sum += 1
    return str(int((sum * 100) / count)) + '%'
