import keras.layers as layer


def train_datal(model_keras, input_dim, train_x, train_y, test_x, test_y):
    model = model_keras.Sequential()
    model.add(layer.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layer.Dense(128, activation='relu'))
    model.add(layer.Dense(128, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(64, activation='relu'))
    model.add(layer.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    history = model.fit(train_x, train_y, epochs=25, batch_size=50, validation_split=0.3)
    accuracy = model.evaluate(test_x, test_y)
    epochs = history.history['val_loss'].__len__()
    print('error percentage', 100 - (accuracy[1] * 100))
    return model, history, epochs
