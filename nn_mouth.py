from scipy import ndimage
import glob
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split



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


X = []
y = []

with open("box_coordinates.csv") as f:
	for line in f:
		data = line.split(",")
		y.append(np.asarray([int(x) for x in data[1:]]))
		image = np.memmap("data/BioID/"+data[0], dtype=np.uint8, shape=(384, 286, 1))
		X.append(image / 255.0)

indices = range(len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.15, random_state=37)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten
from keras.layers.core import Dense, Activation

image = Input(shape=(384, 286, 1))
y = Convolution2D(32, 3, 3, border_mode='same')(image)
z = Dense(8)(Flatten()(y))
model = Model(image, z)

model.compile(loss='mse', optimizer='rmsprop')


model.fit(X_train, y_train, nb_epoch=10, batch_size=20, verbose=1)
np.set_printoptions(threshold=np.nan)
print(model.predict(X_test, batch_size = 16, verbose=1))

check_predictions(predictions, indices_test)




