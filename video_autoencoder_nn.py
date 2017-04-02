def pad_video_features(video, max_length):
    FEATURE_LENGTH = 32

    if len(video) > max_length:
        return video[:max_length]

    while len(video) != max_length:
        image = np.asarray([0]*FEATURE_LENGTH)
        # print(image.shape)
        video.append(image)

    return video

def load_image_features():
    X = []
    y = []

    for folder in glob.glob("videos/P00*/*"):
        video = []
        for file in glob.glob(folder+"/*.bmp"):
            img = mpimg.imread(file)

            features = autoencoder.predict(np.array([img]))
            video.append(features.flatten())

        if video:
            if len(video) != 20:
                video = pad_video_features(video)

            video = np.asarray(video)

            y.append(get_image_class(file))
            X.append(video)

    return X, y

def get_image_class(filename):
    pattern = re.compile(".*\/.*\/([A-z]*)[0-9]*\/[0-9]*.bmp")

    phrase = pattern.match(filename).groups()[0]

    cl = [0]*10
    cl[d[phrase]] = 1

    return cl
    
