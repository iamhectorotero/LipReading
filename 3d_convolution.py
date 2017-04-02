import matplotlib.image as mpimg
from keras.layers import Flatten, Convolution3D, MaxPooling3D
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
import glob, re

d = {"Excuse":0, "Goodby": 1, "Hello": 2, "How": 3, "Nice": 4, "Seeyou": 5, "Sorry": 6, "Thank": 7, "Thanks":7, "Time" : 8, "Welcome": 9}


def get_image_class(filename):
    pattern = re.compile(".*\/.*\/([A-z]*)[0-9]*\/[0-9]*.bmp")

    phrase = pattern.match(filename).groups()[0]

    cl = [0]*10
    cl[d[phrase]] = 1

    return cl

def pad_video(video, max_length=20):

    if len(video) >= max_length:
        return video[:max_length]

    while len(video) != max_length:
        image = np.asarray([[[0]*3]*720]*576)
        video.append(image)

    return video

def load_lip_images(batch_size=4):
    "Generator that loads the frames from the videos."

    X_train = []
    y_train = []

    folders = glob.glob("videos/P00*/*")

    while True:
        for i, folder in enumerate(folders):
            # print("Folder ", i, end="\r")
            video = []
            for file in glob.glob(folder+"/*.bmp")[:20]:
                img = mpimg.imread(file)
                video.append(img)

            if video:
                video = pad_video(video)
                video = np.asarray(video)

                y_train.append(get_image_class(file))
                X_train.append(video)

            if len(X_train) == batch_size:
                yield(np.asarray(X_train), np.asarray(y_train))
                X_train = []
                y_train = []

def create_3d_conv():
    """Create 3D Convolution model for the videos"""

    model = Sequential()

    model.add(MaxPooling3D((1,2,2), input_shape=(20, 576, 720, 3)))
    model.add(Convolution3D(32, 2, 2, 2, activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Convolution3D(16, 2, 2, 2, activation='relu'))
    model.add(MaxPooling3D((2, 2, 3)))
    model.add(Convolution3D(8, 2, 2, 2, activation='relu'))
    model.add(MaxPooling3D((2, 2, 3)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])
    return model

if __name__ == '__main__':

    n_folds = 9
    from sklearn.model_selection import KFold

    skf = KFold(n_splits=n_folds)

    print("Creating Model...")
    model = create_3d_conv()
    model.summary()
    print("Fitting data...")
    model.fit_generator(load_lip_images(), samples_per_epoch = 500, nb_epoch = 10, verbose=1)
    model.evaluate_generator(load_lip_images(), 200)
