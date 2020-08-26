import keras.layers as layer
import keras.models as models


def train_model(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layer.Dense(units=64, input_dim=13, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=32, activation='relu'))
    model.add(layer.Dense(units=32, activation='relu'))
    model.add(layer.Dense(units=16, activation='relu'))
    model.add(layer.Dense(units=8, activation='relu'))
    model.add(layer.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.2)
    acc = model.evaluate(x_test, y_test)
    epochs = history.history['val_loss'].__len__()
    print(f'acc: {acc[1] * 100}')
    return model, history, epochs
