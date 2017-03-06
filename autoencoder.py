import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten, MaxPooling2D, UpSampling2D, Dense
from process_images import *
from PIL import Image

X = []
y = []


for file in glob.glob("data/BioID/*.pgm"):
    img = Image.open(file)
    img = imcrop(img, (128, 128))
    img = np.asarray(img)
    img = np.reshape(img, (128, 128, 1))
    X.append(img / 255.0)
    y.append(img / 255.0)

indices = range(len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', activation="relu", input_shape=(128, 128, 1)))
model.add(MaxPooling2D((4, 4), border_mode='same'))

model.add(Convolution2D(16, 3, 3, border_mode='same', activation="relu"))
model.add(UpSampling2D((4, 4)))
model.add(Convolution2D(1, 3, 3, border_mode='same', activation="sigmoid"))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=10, batch_size=10, verbose=1)
results = model.predict(X_test, batch_size = 16, verbose=1)

for img in results[:4]:
    img = np.reshape(img, (128, 128))
    plt.imshow(img, cmap='gray')
    plt.show()
