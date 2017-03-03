import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense


X = []
y = []

for file in glob.glob("data/BioID/*.pgm"):
    img = mpimg.imread(file)
    img = np.reshape(img, (109824)) / 255.0
    X.append(img)
    y.append(img)

indices = range(len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

model = Sequential()
model.add(Dense(512, input_shape = (109824,)))
model.add(Dense(32))
model.add(Dense(512))
model.add(Dense(109824))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=10, batch_size=20, verbose=1)
results = model.predict(X_test, batch_size = 16, verbose=1)

for img in results[:4]:
    img = np.reshape(img, (286, 384)) * 255.0
    plt.imshow(img, cmap='gray')
    plt.show()
