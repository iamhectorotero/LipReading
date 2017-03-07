import glob, re
import soundfile as sf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Convolution2D, Input, MaxPooling2D, Activation, Reshape
from keras.regularizers import l2, activity_l2
from keras.regularizers import l1, activity_l1
from keras.layers.noise import GaussianNoise

from keras.preprocessing import sequence
import numpy as np
from math import sqrt

from scipy.io.wavfile import read as wavread

def pad(data, length):

    if len(data) > length:
        return data[:length]

    aux = [0]*(length - len(data))
    data = np.append(data, np.asarray(aux))
    return data

def load_data():
    pattern = re.compile(".*/([A-z]*).*")
    d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":8, "Time" : 9, "Welcome": 10}

    X = []
    y = []

    audio_files = glob.glob("videos/*/*/*.wav")
    for i, audio in enumerate(audio_files):
        print("{0:.2f}".format(float(i)*100/len(audio_files)), end="\r")

        phrase = pattern.match(audio).groups()[0]

        if phrase != "Left" and phrase != "Right":
            data, _ = sf.read(audio)

            # [samplerate, data] = wavread(audio)
            data = np.asarray(pad(data, 10000))
            data = data.reshape(1, -1)
            X.append(data)
            cl = [0]*11
            cl[d[phrase]] = 1
            y.append(cl)

            # noise = np.random.normal(0,1,9600)
            #
            # X.append(data*noise)
            # y.append(cl)

    return np.asarray(X), np.asarray(y)

def create_model():
    model = Sequential()
    model.add(SimpleRNN(100, activation="relu", W_regularizer=l2(0.01), return_sequences=True, input_shape=(1, 10000)))
    model.add(Dropout(0.4))
    model.add(SimpleRNN(32,  activation="relu"))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, nb_epoch=25, batch_size=20, verbose=2)
    results = model.evaluate(X_test, y_test, batch_size=20, verbose=1)
    print(results)

if __name__ == '__main__':

    n_folds = 9
    from sklearn.model_selection import KFold

    skf = KFold(n_splits=n_folds)

    print("Loading data...")
    X, y = load_data()

    # sequence.pad_sequences(X, padding="post", maxlen=10000, truncating="post")

    print("Splitting data...")
    for i, (train, test) in enumerate(skf.split(X, y)):
        print("Running Fold "+str(1+i)+"/"+str(n_folds))
        model = create_model()
        model.summary()
        train_and_evaluate(model, X[train], y[train], X[test], y[test])


# model.add(Reshape((38400, 1), input_shape=(1, 38400)))
# model.add(Convolution1D(32, 2))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(8))
# model.add(Convolution1D(16, 2))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(8))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(11, activation='softmax'))
