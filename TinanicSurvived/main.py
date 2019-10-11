from TinanicSurvived import GetData
from TinanicSurvived import TrainData
from TinanicSurvived import DataAnalysis as da
from TinanicSurvived import PredictionModel


def main():
    epochs = 389
    features = ['Pclass', 'Fare', 'Sex', 'Survived']
    feature_len = features.__len__() - 1
    train_x, train_y, test_x, test_y = GetData.get_data(percent=75, features=features)
    print(train_x)
    model, history = TrainData.train_data(train_x=train_x,
                                          train_y=train_y,
                                          test_x=test_x,
                                          test_y=test_y,
                                          epochs=epochs,
                                          input_dim=feature_len)
    da.evaluate_best_epoch(history, epochs)
    print(PredictionModel.prediction_model(model, 800, feature_len, features=features))


if __name__ == '__main__':
    main()
