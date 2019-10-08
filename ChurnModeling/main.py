from ChurnModeling import GetData
from ChurnModeling import TrainData
from ChurnModeling import PredictionModel
from ChurnModeling import DataAnalysis
import keras.models as model


def main():
    features = ['NumOfProducts', 'HasCrCard', 'Balance', 'Tenure', 'Age', 'CreditScore',
                'IsActiveMember',
                'EstimatedSalary',
                'Exited']
    train_x, train_y, test_x, test_y = GetData.get_data(
        features=features,
        percent=75,
        index_feature=(features.__len__() - 1))
    model_trained, model_history, epochs = TrainData.train_datal(model_keras=model,
                                                                 input_dim=(features.__len__() - 1),
                                                                 train_x=train_x,
                                                                 train_y=train_y,
                                                                 test_x=test_x,
                                                                 test_y=test_y)
    print(PredictionModel.prediction_model(
        model=model_trained,
        count=2000,
        features=features,
        index_feature=(features.__len__() - 1)))

    DataAnalysis.evaluate_best_epoch(history=model_history, epochs=epochs)


if __name__ == '__main__':
    main()
