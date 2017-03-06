import glob, re
import soundfile as sf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.preprocessing import sequence
import numpy as np
from math import sqrt

d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Left": 4, "Nice": 5, "Right": 6, "Seeyou": 7, "Sorry": 8, "Thank": 9, "Thanks":9, "Time" : 10, "Welcome": 11 }

pattern = re.compile(".*/([A-z]*).*")

X = []
y = []

for audio in glob.glob("videos/*/*/*.flac"):
    data, _ = sf.read(audio)
    data = np.asarray(data).reshape(-1, 1)
    X.append(data)
    cl = [0]*12
    cl[d[pattern.match(audio).groups()[0]]] = 1
    y.append(cl)




indices = range(len(X))
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.15, random_state=37)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

model = Sequential()
model.add(Convolution1D(32, 3, border_mode='same', input_shape=(200000, 1)))
model.add(MaxPooling1D(4, 4, border_mode='same'))
model.add(Convolution1D(16, 3, border_mode='same'))
model.add(MaxPooling1D(4, 4, border_mode='same'))
model.add(Convolution1D(4, 3, border_mode='same'))
model.add(MaxPooling1D(4, 4, border_mode='same'))
model.add(Convolution1D(1, 3, border_mode='same'))
model.add(MaxPooling1D(4, 4, border_mode='same'))
model.add(Flatten())

# model.add(Dropout(0.2))
# model.add(SimpleRNN(100))
# model.add(Dropout(0.2))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

print("Padding sequences...")
X_train = sequence.pad_sequences(X_train, padding='post', maxlen=200000)
X_test = sequence.pad_sequences(X_test, padding='post', maxlen=200000)

print("Training...")
model.fit(X_train, y_train, nb_epoch=10, batch_size=20, verbose=1)
