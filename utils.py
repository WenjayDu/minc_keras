import os
import sys
from os.path import splitext, basename

import h5py
import numpy as np

global categorical_functions
categorical_functions = ["categorical_crossentropy"]


def from_categorical(cat, img):
    out = np.zeros(img.shape)
    for i, cat0 in zip(np.unique(img), cat):
        out = out + cat0 * i
    return (out)


def set_model_name(filename, target_dir, ext='.hdf5'):
    '''function to set default model name'''
    return target_dir + os.sep + splitext(basename(filename))[0] + ext


def safe_h5py_open(filename, mode):
    '''open hdf5 file, exit elegantly on failure'''
    # meera
    # At the moment, this function returns a complicated object "f" that contains
    # the image array somewhere inside of it. 
    # You can modify this function so that it uses nibabel to load in images instead
    # of h5py. In this case, this function should return the actual 3D/4D array.
    # 
    try:
        # meera
        # not sure if this is right, but could try something like :
        # f = nibabel.Load(filename)
        # image_array = np.asarray(f.dataobj)
        # return image_array
        f = h5py.File(filename, mode)
        return f

    except OSError:
        print('Error: Could not open', filename)
        sys.exit(1)


def normalize(np_array):
    """
    performs a simple normalization from 0 to 1 of a numpy array.
    checks that the image is not a uniform value first

    :param np_array: numpy array
    :return: numpy array (either A or normalized version of A)
    """
    std_factor = 1
    # cal the global standard deviation
    global_std_deviation = np.std(np_array)
    if global_std_deviation > 0:
        std_factor = global_std_deviation
    np_array = (np_array - np.mean(np_array)) / std_factor

    scale_factor = np.max(np_array) - np_array.min()
    if scale_factor == 0:
        scale_factor = 1
    np_array = (np_array - np_array.min()) / scale_factor
    return np_array
