import pandas
import glob
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Input
from keras.layers.convolutional import Convolution2D,Convolution3D

import tflearn


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

dataframe = pandas.read_csv("videos/data.csv", header=None)
y = dataframe.values.astype(float)

X = []

for folder in glob.glob("videos/*"):
    video = []
    for file in glob.glob(folder+"/*.bmp"):
        img = mpimg.imread(file)
        video.append(np.reshape(rgb2gray(img), (921600)))
    if video:
        X.append(video)

indices = range(len(X))
#
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.5, random_state=37)
#
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print(X_train.shape)

model = Sequential()
model.add(SimpleRNN(32, input_shape=(100, 921600)))
model.add(Dense(4))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=10, batch_size=20, verbose=1)
model.evaluate(X_test, y_test, batch_size=16)


# Save it.
# model.save('checkpoints/rnn.tflearn')
