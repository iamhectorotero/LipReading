import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from random import shuffle

BOX_COORDINATES_CSV = "box_coordinates.csv"

#Read the values in the CSV and plot the boxes with matplotlib 

with open(BOX_COORDINATES_CSV) as f:

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

        corners = [y, y+width, y+width, y, y], [x, x, x+height, x+height, x]

        plt.plot(corners)
        plt.show()
