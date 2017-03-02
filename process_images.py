import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, sys, getopt
from PIL import Image

def impad(img):
    '''
    Pads an image to create a square
    INPUT/OUTPUT: PIL.Image
    '''
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2

    res = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )
    )
    return res

def imcrop(img, size):
    '''
    Crops an image of specified size in the center
    INPUT: PIL.Image, tuple = (width, height)
    OUTPUT: PIL.Image
    DOES NOT WORK ATM
    '''
    # half_the_width = img.size[0] / 2
    # half_the_height = img.size[1] / 2
    # w = size[0] / 2
    # h = size[1] / 2
    #
    # print half_the_width, half_the_height, w, h
    # print half_the_width - w, half_the_height - h, half_the_width + w, half_the_height + h

    width, height = img.size   # Get dimensions
    new_width, new_height = size

    # left = (width - new_width)/2
    # top = (height - new_height)/2
    # right = (width + new_width)/2
    # bottom = (height + new_height)/2
    #
    # im.crop((left, top, right, bottom))

    res = img.crop(
        (
            (width - new_width) / 2,
            (height - new_height) / 2,
            (width + new_width) / 2,
            (height + new_height) / 2
        )
    )

    # res = img.crop(
    #     (
    #         half_the_width - w,
    #         half_the_height - h,
    #         half_the_width + w,
    #         half_the_height + h
    #     )
    # )
    return res

def imresize(img, size):
    '''
    Rescales an image to desired size
    INPUT: PIL.Image, tuple = (width, height)
    OUTPUT: PIL.Image
    '''
    res = img.resize(size)
    return res

def process_images(in_fold, out_fold, s, m):
    size = (s,s)
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    for dr in os.listdir(in_fold):
        print ">> Processing images in", dr, "folder"
        if not os.path.exists(out_fold + '/' + dr):
            os.makedirs(out_fold + '/' + dr)
        for f in os.listdir(in_fold + '/' + dr):
            if not f.endswith(".pgm") and not f.endswith(".jpg"):
                continue
            img = Image.open(in_fold + '/' + dr + '/' + f)
            if m == 'resize':
                img = imresize(impad(img), size)
            # else:
                # img = imcrop(img, size) TODO: FIX!
            img.save(out_fold + '/' + dr + '/' + f)


def main(argv):
    input_folder = 'data'
    output_folder = 'processed_data'
    size = 100
    mode = 'resize'
    try:
      opts, args = getopt.getopt(argv,"hi:o:s:m:")
    except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <input_folder> -o <output_folder> -s <size> -m <crop/resize>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
        elif opt in ("-s"):
            size = int(arg)
        elif opt in ("-m"):
            if arg in ['resize', 'crop']:
                mode = arg
            else:
                print 'Invalid mode of operation'
                print 'test.py -i <input_folder> -o <output_folder> -s <size> -m <crop/resize>'
                sys.exit()
    process_images(input_folder, output_folder, size, mode)

if __name__ == "__main__":
   main(sys.argv[1:])
