import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.columns = ['label', 'text']
labels = np.where(df['label'] == 'spam', 1, 0)
X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                    labels, test_size=0.2, random_state=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
x_train_seq = tokenizer.texts_to_sequences(X_train)
x_test_seq = tokenizer.texts_to_sequences(X_test)
x_train_padded = pad_sequences(x_train_seq, 50)
x_test_padded = pad_sequences(x_test_seq, 50)

model = Sequential()
model.add(Embedding(len(tokenizer.index_word) + 1, 32))
model.add(LSTM(32, dropout=0, recurrent_dropout=0))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
model.fit(x_train_padded, y_train, validation_data=(x_test_padded, y_test), epochs=10, batch_size=32)
history = model.history
plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], label='Accuracy of Train')
plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], label='Accuracy of Validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('my first lstm model for spam detection')
plt.legend()
plt.show()
