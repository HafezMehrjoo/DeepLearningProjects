import keras.models as models
import keras.layers as layer


def train_data(train_x, test_x, train_y, test_y, input_dim, epochs):
    model = models.Sequential()
    model.add(layer.Dense(units=64, input_dim=input_dim, activation='relu'))
    model.add(layer.Dense(units=32, activation='relu'))
    model.add(layer.Dense(units=32, activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(rate=0.3))
    model.add(layer.Dense(units=16, activation='relu'))
    model.add(layer.Dense(units=16, activation='relu'))
    model.add(layer.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=10, validation_split=0.2)
    acc = model.evaluate(test_x, test_y)
    print('Error Percentage', 100 - (acc[1] * 100))
    return model, history
