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
from keras.layers.convolutional import Convolution2D
import pandas

from random import Random

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Left": 4, "Nice": 5, "Right": 6, "Seeyou": 7, "Sorry": 8, "Thank": 9, "Time" : 10, "Welcome": 11 }

folders = glob.glob("videos/P00*/*")
Random(4).shuffle(folders)

training_folders = folders[:-64]
test_folders = folders[-64:]

model = Sequential()
model.add(SimpleRNN(32, input_shape=(10, 414720)))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32

def generate_video_array():
    X_train = []
    y_train = []

    while True:
        for folder in training_folders:
            video = []
            classes = [0]*12
            for file in glob.glob(folder+"/*.bmp")[:10]:
                img = mpimg.imread(file)
                video.append(np.reshape(rgb2gray(img), (414720, 3)))

            if video:
                classes[d[folder.split("/")[-1][:-1]]] = 1
                y_train.append(np.asarray(classes))
                X_train.append(np.asarray(video))

            if len(X_train) == batch_size:
                yield(np.asarray(X_train), np.asarray(y_train))
                X_train = []
                y_train = []

model.fit_generator(generate_video_array(), samples_per_epoch=batch_size*16, nb_epoch=10, verbose=1)

X_test = []
y_test = []

for folder in folders[-64:]:
    video = []
    classes = [0]*12
    for file in glob.glob(folder+"*.bmp"):
        img = mpimg.imread(file)
        video.append(np.reshape(rgb2gray(img), (414720)))

    if video:
        X_test.append(video)
        classes[d[folder.split("/")[-1][:-1]]] = 1
        y_test.append(classes)


model.evaluate(X_test, y_test, batch_size=16)


# Save it.
# model.save('checkpoints/rnn.tflearn')
