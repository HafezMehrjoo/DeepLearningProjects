from HeartDiseaseUCI import GetData
from HeartDiseaseUCI import TrainData
from HeartDiseaseUCI import PredictionModel
from HeartDiseaseUCI import DataAnalysis
import keras.models as model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = GetData.get_data(percent=90)
    model_trained, model_history, epochs = TrainData.train_model(model, train_x, train_y, test_x, test_y)
    print(PredictionModel.prediction_model(model=model_trained, count=303))
    DataAnalysis.evaluate_best_epoch(history=model_history, epochs=epochs)
