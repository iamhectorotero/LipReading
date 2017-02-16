from scipy import ndimage
import glob
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


X = []
y = []


with open("box_coordinates.csv") as f:
	for line in f:
		data = line.split(",")
		y.append(np.asarray([int(x) for x in data[1:]]))
		X.append(np.memmap("data/"+data[0], dtype=np.uint8))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(8, activation='linear', input_shape=(1591296,)))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=15, batch_size=20, verbose=1)
score = model.evaluate(X_test, y_test, batch_size=16, verbose=1)

print(score)


# (X_train, y_train), (X_test, y_test) = mnist.load_data()
