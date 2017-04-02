from scipy import ndimage
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from random import shuffle
from keras.callbacks import EarlyStopping

def check_predictions(predictions, indices_test):

	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg

	with open("box_coordinates.csv") as f:
	    photos = f.readlines()

	for i,index in enumerate(indices_test):
	    title = photos[index].split(',')[0]
	    img = mpimg.imread("data/BioID/"+title)
	    simgplot = plt.imshow(img)
	    x_coords = predictions[i][0::2]
	    y_coords = predictions[i][1::2]
	    plt.plot(x_coords, y_coords)
	    plt.show()

from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten, MaxPooling2D
from keras.layers.core import Dense, Activation

print("Compiling model...")

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(288, 360, 3)))
model.add(MaxPooling2D((8,8)))
model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(MaxPooling2D((4,4)))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(4))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')

X = []
y = []

with open("dlib/box_coordinates.csv") as f:
    c = 0
    lines = f.readlines()
    shuffle(lines)
    for line in lines[:20000]:
        c += 1
        print(" ", "{0:.2f}".format(100*float(c)/3000), end="\r")
        data = line.split(",")

        if not "Left" in data[0] and not "Right" in data[0]:
            y.append(np.asarray([float(x) for x in data[1:]]))
            img = Image.open("videos/"+data[0])
            img = img.resize((360, 288), Image.ANTIALIAS)
            img = np.asarray(img)
            X.append(img)
            # print(data, "HWAT")

print(len(X))
indices = range(len(X))
print("Splitting data..")
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.15, random_state=37)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print("Fitting the data...")

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(X_train, y_train, nb_epoch=10, batch_size=20, verbose=1, callbacks=[early_stopping], validation_split=0.1)
np.set_printoptions(threshold=np.nan)
pred = model.predict(X_test, batch_size = 16, verbose=1)

check_predictions(pred, indices_test)
