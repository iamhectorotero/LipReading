#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#To make sure the mouth is inside the bounding box
PIXEL_MARGIN = 20
csv_lines = []

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # print("Processing file: {}".format(f))
    img = mpimg.imread(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    #If more than one face is detected, we skip the image
    if len(dets) != 1:
    	continue

    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
       
       	#For the points that form each landmark, extract the coordinates
       	#of the ones that belong to the mouth (last 20)
        points = [(point.x, point.y) for point in shape.parts()[-20:]]

        #Construct a box out of the the extreme points of the mouth
        upmost = min(points, key = lambda x: x[0])
        downmost = max(points, key = lambda x: x[0])
        leftmost = min(points, key = lambda x: x[1])
        rightmost = max(points, key = lambda x: x[1])
        
        #Horizontal lenght and Vertical length of the box
        hor_var = (rightmost[0] - leftmost[0], rightmost[1] - leftmost[1])
        vert_var = (downmost[0] - upmost[0], downmost[1] - upmost[1])

        #x, y coordinates of each corner
        left_up_corner = (leftmost[0] - vert_var[0]/2 - PIXEL_MARGIN, leftmost[1] - vert_var[1]/2 - PIXEL_MARGIN)
        left_down_corner = (leftmost[0] + vert_var[0]/2 + PIXEL_MARGIN, leftmost[1] + vert_var[1]/2 - PIXEL_MARGIN)
        right_up_corner = (rightmost[0] - vert_var[0]/2 - PIXEL_MARGIN, rightmost[1] - vert_var[1]/2 + PIXEL_MARGIN)
        right_down_corner = (rightmost[0] + vert_var[0]/2 + PIXEL_MARGIN, rightmost[1] + vert_var[1]/2 + PIXEL_MARGIN)

        corners = [left_up_corner, left_down_corner, right_down_corner, right_up_corner, left_up_corner,]

        # Draw the face landmarks on the screen.
        x_coords = [x for x, _ in corners[:4]]
        y_coords = [y for _, y in corners[:4]]

        #Uncomment these lines to show the box plot
        # simgplot = plt.imshow(img)
        # plt.plot(x_coords, y_coords)
        # plt.show()
        photo_name = f.split("/")[-1]
        coords = []
        for x, y in zip(x_coords, y_coords):
        	coords.extend([str(x), str(y)])

        csv_lines.append(photo_name+","+",".join(coords)+"\n")

        
with open("box_coordinates.csv", "a") as csv:
	for line in csv_lines:
		csv.write(line)
        	


