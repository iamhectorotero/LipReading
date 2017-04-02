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

from random import shuffle


folders = shuffle(glob.glob("videos/P00*/*"))
TRAINING_FOLDERS = folders[:-64]
TEST_FOLDERS = folders[-64:]

d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Time" : 8, "Welcome": 9}
BATCH_SIZE = 32

def get_model():
    """Define and compile a model"""
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=(10, 414720)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def get_test_sets():
    """Load test examples and classes"""
    X_test = []
    y_test = []

    for folder in TEST_FOLDERS:
        video = []
        classes = [0]*10

        #Read images for each video, and convert them to grayscale
        for file in glob.glob(folder+"*.bmp"):
            img = mpimg.imread(file)
            video.append(np.reshape(rgb2gray(img), (414720)))

        if video:
            X_test.append(video)
            #Mark the class according to the global dictionary reading the path
            phrase = folder.split("/")[-1][:-1]
            classes[d[phrase]] = 1
            y_test.append(classes)

    return X_test, y_test

def rgb2gray(rgb):
    """Convert RGB values to their corresponding grayscale values"""
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def generate_video_array():
    """Video examples batch generator"""
    X_train = []
    y_train = []

    while True:
        for folder in TRAINING_FOLDERS:
            video = []
            classes = [0]*10

            #Read the images for a video
            for file in glob.glob(folder+"/*.bmp"):
                img = mpimg.imread(file)
                video.append(np.reshape(rgb2gray(img), (414720, 3)))

            if video:
                classes[d[folder.split("/")[-1][:-1]]] = 1
                y_train.append(np.asarray(classes))
                X_train.append(np.asarray(video))

            #If the batch is full, yield it
            if len(X_train) == BATCH_SIZE:
                yield(np.asarray(X_train), np.asarray(y_train))
                X_train = []
                y_train = []

model = get_model()
model.fit_generator(generate_video_array(), samples_per_epoch=BATCH_SIZE*16, nb_epoch=10, verbose=1)

X_test, y_test = get_test_sets()
model.evaluate(X_test, y_test, batch_size=16)
