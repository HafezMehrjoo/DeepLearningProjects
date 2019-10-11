from TinanicSurvived import GetData


def prediction_model(model, count, feature_len, features):
    if count > 891:
        return 'max of count is 891'
    array = GetData.get_data(features=features, predict=True)
    x = array[:, 0:feature_len]
    y = array[:, feature_len:]
    sum = 0
    predictions = model.predict_classes(x)
    for i in range(count):
        if predictions[i] == y[i]:
            sum += 1
    return str(int((sum * 100) / count)) + '%'
