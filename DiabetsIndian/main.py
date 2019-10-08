import keras.models as model
from DiabetsIndian import GetData
from DiabetsIndian import TrainData
from DiabetsIndian import PredictionModel
from DiabetsIndian import DataAnalysis


def main():
    x_train, y_train, x_test, y_test = GetData.get_percent_data(70)
    model_trained, model_history, epochs = TrainData.train_datal(model_keras=model, x_train=x_train, y_train=y_train,
                                                                 x_test=x_test,
                                                                 y_test=y_test)
    print(PredictionModel.prediction_model(model=model_trained, count=767))
    DataAnalysis.evaluate_best_epoch(history=model_history, epochs=epochs)


if __name__ == '__main__':
    main()

