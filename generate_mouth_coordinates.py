import sys
import os
import dlib
import glob
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

predictor_path = sys.argv[1]
PATH = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

csv_lines = []

for f in glob.glob(PATH):
    # print("Processing file: {}".format(f[10:]))
    img = mpimg.imread(f)

    # Ask the detector to find the bounding boxes of each face
    dets = detector(img, 1)

    #If more than one face is detected, we skip the image
    if len(dets) != 1:
    	continue

    for k, d in enumerate(dets):
        # Get the landmarks for the face in box d.
        shape = predictor(img, d)

       	#For the points that form each landmark, extract the coordinates
       	#of the ones that belong to the mouth (last 20).
        #Note: We switch x and y because of the different naming convention for
        #the landmarks and matplotlib
        y_coords = [point.x for point in shape.parts()[-20:]]
        x_coords = [point.y for point in shape.parts()[-20:]]

        #Construct a box out of the the extreme points of the mouth
        #x, y coordinates of each corner
        min_y = min(y_coords)
        max_y = max(y_coords)

        min_x = min(x_coords)
        max_x = max(x_coords)

        height = (max_x - min_x) * 2.1
        width = (max_y - min_y) * 1.6

        corners = [min_x - height * 0.3, min_y - width * 0.2, height, width]

        # Draw the face landmarks on the screen.
        x_coords = [corners[0], corners[0], corners[0] + height, corners[0] + height, corners[0]]
        y_coords = [corners[1], corners[1] + width, corners[1] + width, corners[1], corners[1]]

        #Uncomment these lines to show the box plot
        # simgplot = plt.imshow(img)
        # plt.plot(y_coords, x_coords)
        # plt.show()
        corners = [str(x) for x in corners]
        csv_lines.append(f[10:]+","+",".join(corners)+"\n")


#Write the lines to a CSV file
with open("box_coordinates.csv", "w+") as csv:
	for line in csv_lines:
		csv.write(line)
