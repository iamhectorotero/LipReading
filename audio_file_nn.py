import glob, re
import soundfile as sf
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Flatten, Convolution3D, MaxPooling3D, Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.applications.music_tagger_crnn import MusicTaggerCRNN
from keras.applications.music_tagger_crnn import preprocess_input, decode_predictions

AUDIO_SIZE = 44000
d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":7, "Time" : 8, "Welcome": 9}

def get_class(filename):
    """Given a filename return a hot encoded vector"""

    pattern = re.compile(".*/([A-z]*).*")

    phrase = pattern.match(filename).groups()[0]

    if "_smooth" in phrase:
        phrase = phrase[:-7]

    cl = [0]*10
    cl[d[phrase]] = 1

    return cl

def pad_sound(data, length):

    if len(data) > length:
        return data[:length]

    aux = [0]*(length - len(data))
    data = np.append(data, np.asarray(aux))
    return data

def load_audio_features():
    """Load the audio files and preprocess them to obtain their MusicTaggerCRNN features
    and class into two numpy arrays"""

    X = []
    y = []

    audio_files = glob.glob("videos/*/*/*.wav")
    model = MusicTaggerCRNN(weights='msd', include_top=False)

    for i, audio in enumerate(audio_files):
        print(" {0:.2f}".format(float(i)*100/len(audio_files)), end="\r")

        melgram = preprocess_input(audio)
        melgrams = np.expand_dims(melgram, axis=0)

        feats = model.predict(melgrams)

        X.append(feats)
        y.append(get_class(audio))

    return np.asarray(X), np.asarray(y)

def load_sound_data():
    """Load the audio files and their class into two numpy arrays"""

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

def create_model_mtc(trainable=False):
    """Create a model that receives as input the audio features by the MusicTaggerCRNN.
    If trainable, the base structure is trained."""

    mtc = MusicTaggerCRNN(weights='msd', include_top=False)
    mtc.trainable = trainable

    model = Sequential()
    model.add(mtc)
    model.add(Dense(10, activation='softmax', input_shape=(1, 32)))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def create_sound_conv():
    """Create a 1D convolution model that receives as input the audio files"""
    model = Sequential()
    model.add(Convolution1D(32, 3, activation='relu', input_shape=(AUDIO_SIZE,1)))
    model.add(MaxPooling1D(8))
    model.add(Convolution1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, y_train, nb_epoch=25, batch_size=10, verbose=1, validation_split=0.1, callbacks=[early_stopping])
    results = model.evaluate(X_test, y_test, batch_size=20, verbose=1)
    print(results)

if __name__ == '__main__':

    # print("Loading data...")
    # X,y = load_sound_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #
    # print("Creating Model...")
    # model = create_sound_conv()
    # model.summary()
    #
    # train_and_evaluate(model, X_train, y_train, X_test, y_test)

    print("Loading data...")
    X,y = load_audio_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("Creating Model...")
    model = create_model_mtc()
    model.summary()

    train_and_evaluate(model, X_train, y_train, X_test, y_test)
