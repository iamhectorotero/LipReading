import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from random import shuffle
with open("dlib/box_coordinates.csv") as f:
    c = 0
    lines = f.readlines()
    shuffle(lines)
    for line in lines:
        name, x, y, height, width = line.split(",")
        img = mpimg.imread("videos/"+name)
        plt.imshow(img)

        x = float(x)
        y = float(y)
        width = float(width)
        height = float(height)
        print([y, y+width, y+width, y, y], [x, x, x+height, x+height, x])

        plt.plot([y, y+width, y+width, y, y], [x, x, x+height, x+height, x])
        plt.show()
