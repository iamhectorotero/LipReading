# LipReading

Neural Network project to caption videos obtained from OuluVS database. 

# File Description

* 3d_convolution.py: Predict video class using 3D Convolution directly on the videos.

* flattened_video_rnn.py: Predict video class by using a sequence of flattened images.

* audio_file_nn.py: Predict video class using the audio files.

* audio_plot_nn.py: Predict video class using the plot of the audio files.

* extract_audio_files.py: Takes a list of videos and extracts mono audio (WAV) files.

* plot_bounding_box.py: Read the values in the CSV and plot the boxes with matplotlib.

* generate_mouth_coordinates.py: Generate the CSV file for the mouth bounding box.

* process_images.py: Methods to reduce the resolution or properties of images.

# Requirements to generate box_coordinates.csv

* Dlib

* ffmpeg

* Numpy

* Matplotlib
