import glob, re
import soundfile as sf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Dropout, Masking
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Convolution2D, Input, MaxPooling2D, Activation, Reshape
from keras.regularizers import l2, activity_l2
from keras.regularizers import l1, activity_l1

import matplotlib.image as mpimg
from keras.callbacks import EarlyStopping


from keras.preprocessing import sequence
import numpy as np
from math import sqrt

from keras.applications.music_tagger_crnn import MusicTaggerCRNN
from keras.applications.music_tagger_crnn import preprocess_input, decode_predictions

from keras.applications.vgg16 import VGG16

vgg16 = VGG16(include_top=False, input_shape=(576, 720, 3))
autoencoder = load_model('saved_autoencoder/autoencodermodel')
autoencoder.load_weights('saved_autoencoder/autoencoder_weights.h5')

AUDIO_SIZE = 44000

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_class(filename):
    pattern = re.compile(".*/([A-z]*).*")
    d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":8, "Time" : 9, "Welcome": 10}

    phrase = pattern.match(filename).groups()[0]

    cl = [0]*11
    cl[d[phrase]] = 1

    return cl

def get_image_class(filename):
    pattern = re.compile(".*\/.*\/([A-z]*)[0-9]*\/[0-9]*.bmp")
    d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":8, "Time" : 9, "Welcome": 10}
    # print(filename)

    phrase = pattern.match(filename).groups()[0]

    cl = [0]*11
    cl[d[phrase]] = 1

    return cl

def pad_sound(data, length):

    if len(data) > length:
        return data[:length]

    aux = [0]*(length - len(data))
    data = np.append(data, np.asarray(aux))
    return data

def pad_video(video, max_length):

    if len(video) > max_length:
        return video[:max_length]

    while len(video) != max_length:
        image = np.asarray([0]*1244160)
        # print(image.shape)
        video.append(image)

    return video
def pad_video_features(video, max_length):
    FEATURE_LENGTH = 32

    if len(video) > max_length:
        return video[:max_length]

    while len(video) != max_length:
        image = np.asarray([0]*FEATURE_LENGTH)
        # print(image.shape)
        video.append(image)

    return video

def load_audio_features():
    X = []
    y = []

    for i, audio in glob.glob("videos/*/*/*.wav"):
        print(" {0:.2f}".format(float(i)*100/len(audio_files)), end="\r")

        melgram = preprocess_input(audio)
        melgrams = np.expand_dims(melgram, axis=0)

        X.append(feats)
        y.append(get_class(audio))

    return np.asarray(X), np.asarray(y)

def load_image_features():
    X = []
    y = []

    for folder in glob.glob("videos/P00*/*"):
        video = []
        for file in glob.glob(folder+"/*.bmp"):
            img = mpimg.imread(file)

            if crop_lips:
                img = crop_lips(img)
            features = vgg16.predict(np.array([img]))
            # print(features.flatten().shape)
            video.append(features.flatten())

        if video:
            if len(video) != max_length:
                video = pad_video_features(video, max_length)

            video = np.asarray(video)

            y.append(get_image_class(file))
            X.append(video)

    return X, y

def load_plot_images():
    X = []
    y = []

    image_files = glob.glob("videos/*/*/*.png")

    for i, image in enumerate(image_files):
        print(" {0:.2f}".format(float(i)*100/len(image_files)), end="\r")

        img = mpimg.imread(image)
        img = rgb2gray(img)
        img = img.reshape(640, 480, 1)

        X.append(img)
        y.append(get_class(image))

    return np.asarray(X), np.asarray(y)

def load_sound_data():

    X = []
    y = []

    audio_files = glob.glob("videos/*/*/*.wav")
    for i, audio in enumerate(audio_files):
        print(" {0:.2f}".format(float(i)*100/len(audio_files)), end="\r")

        data, _ = sf.read(audio)
        data = pad_sound(data, AUDIO_SIZE)
        data = data.reshape(-1, 1)

        X.append(data)
        y.append(get_class(audio))

    return np.asarray(X), np.asarray(y)

def load_lip_images(batch_size, max_length, crop_lips=False):
    X_train = []
    y_train = []

    folders = glob.glob("videos/P00*/*")

    while True:
        for folder in folders:
            video = []
            for file in glob.glob(folder+"/*.bmp"):
                img = mpimg.imread(file)

                if crop_lips:
                    img = crop_lips(img)

                img = img.reshape(-1,)
                # print(img.shape)
                video.append(img)

            if video:
                if len(video) != max_length:
                    video = pad_video(video, max_length)

                video = np.asarray(video)

                # print(video.shape)

                y_train.append(get_image_class(file))
                X_train.append(video)
                # print(len(X_train))

            if len(X_train) == batch_size:
                yield(np.asarray(X_train), np.asarray(y_train))
                X_train = []
                y_train = []

def crop_lips(img):
    return img
def create_model_mtc(trainable=False):
    mtc = MusicTaggerCRNN(weights='msd', include_top=False)
    mtc.trainable = trainable

    model = Sequential()
    model.add(mtc)
    model.add(Dense(11, activation='softmax', input_shape=(1, 32)))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def create_sound_conv():
    model = Sequential()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def create_plot_conv():
    model = Sequential()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def create_lip_conv():

    model = Sequential()

    model.add(SimpleRNN(11, activation='softmax', input_shape=(30, 1244160)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.fit(X_train, y_train, nb_epoch=25, batch_size=10, verbose=1, validation_split=0.1, callbacks=[early_stopping])
    results = model.evaluate(X_test, y_test, batch_size=20, verbose=1)
    print(results)

if __name__ == '__main__':

    n_folds = 9
    from sklearn.model_selection import KFold

    skf = KFold(n_splits=n_folds)

    print("Loading data...")
    X, y = load_image_features()

    # sequence.pad_sequences(X, padding="post", maxlen=10000, truncating="post")

    print("Splitting data...")
    for i, (train, test) in enumerate(skf.split(X, y)):
        print("Running Fold "+str(1+i)+"/"+str(n_folds))
        model = create_lip_conv()
        model.summary()
        train_and_evaluate(model, X[train], y[train], X[test], y[test])
    # model.fit_generator(load_lip_images(1, 30), samples_per_epoch=800, nb_epoch=10)
    # model.evaluate_generator(load_lip_images(5, 30), 200)
