from HeartDiseaseUCI import GetData
from HeartDiseaseUCI import TrainData
from HeartDiseaseUCI import DataAnalysis
import keras.models as model

if __name__ == '__main__':
    x_train, x_test,y_train , y_test = GetData.get_data(percent=90)
    model_trained, model_history, epochs = TrainData.train_model(x_train, y_train, x_test, y_test)
