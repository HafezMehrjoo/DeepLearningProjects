import keras.layers as layer


def train_model(kereas_model, train_x, train_y, test_x, test_y):
    model = kereas_model.Sequential()
    model.add(layer.Dense(units=64, input_dim=13, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=64, activation='relu'))
    model.add(layer.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    history = model.fit(train_x, train_y, epochs=25, batch_size=1, validation_split=0.2)
    acc = model.evaluate(test_x, test_y)
    epochs = history.history['val_loss'].__len__()
    print('Error Percentage', 100 - (acc[1] * 100))
    return model, history, epochs
