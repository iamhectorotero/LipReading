from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Flatten, Convolution2D
import glob, re
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":7, "Time" : 8, "Welcome": 9}

def get_class(filename):
    """Given a filename return a hot encoded vector of the class"""

    pattern = re.compile(".*/([A-z]*).*")

    phrase = pattern.match(filename).groups()[0]

    if "_smooth" in phrase:
        phrase = phrase[:-7]

    cl = [0]*10
    cl[d[phrase]] = 1

    return cl

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_plot_conv():
    model = Sequential()
    model.add(Convolution2D(32, 2, 2, input_shape=(640, 480, 1)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, y_train, nb_epoch=25, batch_size=10, verbose=1, validation_split=0.1, callbacks=[early_stopping])
    results = model.evaluate(X_test, y_test, batch_size=20, verbose=1)
    print(results)

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


if __name__ == '__main__':

    print("Loading data...")
    X,y = load_plot_images()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("Creating Model...")
    model = create_plot_conv()
    model.summary()

    train_and_evaluate(model, X_train, y_train, X_test, y_test)
