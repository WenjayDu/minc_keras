import numpy as np
import scipy as sp
import pandas as pd
import h5py
#from pyminc.volumes.factory import *
import os
from re import sub
from sys import argv, exit
from os.path import basename, exists, splitext
from os import makedirs
from set_images import *
from utils import *

def adjust_batch_size(n1, n2, batch_size):
    #This little bit of code changes the batch_size so that it divides the first dimension
    #of the data tensor without remainder. This way the data tensor can be divided into 
    #equally sized batche
    n = n1
    if n > n2:n=n2
    if n1 % batch_size != 0 and n2 % batch_size != 0:
        for b in range(n, 0, -1):
            if n1 % b == 0 and n2 % b == 0:
                return b
    else: return batch_size


def feature_extraction(images,image_dim, x_output_file, y_output_file,target_dir, clobber):
    nSubjects= images.shape[0] #total number f subjects
    total_valid_slices = images.valid_samples.values.sum()
    #Set up the number of valid slices for each subject

    f = h5py.File(target_dir+os.sep+'temp.hdf5', "w")
    X_f = f.create_dataset("image", [total_valid_slices,image_dim[1],image_dim[2],1], dtype='float16')
    Y_f = f.create_dataset("label", [total_valid_slices,image_dim[1],image_dim[2],1], dtype='float16')
    total_index=0
    for index, row in images.iterrows():
        if index % 10 == 0: print("Saving",images["category"][0],"images:",index, '/', images.shape[0] , end='\r') 
        minc_pet_f = safe_h5py_open(row.pet, 'r')
        minc_label_f = safe_h5py_open(row.label, 'r')
        pet=np.array(minc_pet_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.pet).data

        #sum the pet image if it is a 4d volume
        if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
        label=np.array(minc_label_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.label).data
        pet = normalize(pet) # (pet - pet.min())/(pet.max() - pet.min())
        pet=pet.reshape(list(pet.shape)+[1])

        label = normalize(label) #(label - label.min())/(label.max() - label.min())
        label=label.reshape(list(label.shape)+[1])
        for j in range(row.total_samples):
            if pet[j].sum() != 0 : 
                f['image'][(total_index)] = pet[j]
                f['label'][(total_index)] = label[j]
                total_index += 1

    clean_X = f['image']#[not_index_bad_X]
    clean_Y = f['label']#[not_index_bad_X]
    np.save(x_output_file,clean_X)
    np.save(y_output_file,clean_Y)
    f.close()
    print("")
    return( 0  )



def set_onehot(images, filename):
    onehot = np.array([])
    for i, nsamples in zip(images.onehot, images.valid_samples): 
        onehot=np.concatenate([onehot, np.repeat(i, nsamples)] )
    np.save(filename, onehot)
    return(0)

def get_image_dim(fn):
    '''get spatial dimensions for input images
    fn -- filename 
    '''
    minc_label_f = safe_h5py_open(fn, 'r')
    label_img = np.array(minc_label_f['minc-2.0/']['image']['0']['image'])
    image_dim = list(label_img.shape) #load label file and get its dimensions
    del label_img
    return image_dim

# Go to the source directory and grab the relevant data. Convert it to numpy arrays named validate- and train-
def prepare_data(source_dir, target_dir, input_str, label_str, ratios, batch_size, feature_dim=2, images_fn='images.csv', onehot_label=None, clobber=False):
    ### 0) Setup file names and output directories
    data_dir = target_dir + os.sep + 'data' + os.sep
    report_dir = target_dir+os.sep+'report'
    if not exists(data_dir): makedirs(data_dir)
    if not exists(report_dir): makedirs(report_dir) 

    prepare_data.train_x_fn = data_dir + os.sep + 'train_x'
    prepare_data.train_onehot_fn = data_dir + os.sep + 'train_onehot'
    prepare_data.train_y_fn = data_dir + os.sep + 'train_y'
    prepare_data.validate_x_fn = data_dir + os.sep + 'validate_x'
    prepare_data.validate_onehot_fn = data_dir + os.sep + 'validate_onehot'
    prepare_data.validate_y_fn = data_dir + os.sep + 'validate_y'


    ### 1) Organize inputs into a data frame, match each PET image with label image
    images_fn = set_model_name(images_fn, report_dir, '.csv')
    if not exists(images_fn) or clobber: images = set_images(source_dir, target_dir, ratios,images_fn, input_str, label_str )
    else: images = pd.read_csv(images_fn)
    ## 1.5) Split images into training and validate data frames
    train_images = images[images['category']=='train'].reset_index()
    validate_images = images[images['category']=='validate'].reset_index()
    train_valid_samples = train_images.valid_samples.values.sum()  
    validate_valid_samples  =  validate_images.valid_samples.values.sum()

    ### 2) Get spatial dimensions of images 
    image_dim = get_image_dim(images.iloc[0].label)

    ### 3) Set up dimensions of data tensors to be used for training and validateing. all of the


    if not exists(prepare_data.train_x_fn + '.npy') or not exists(prepare_data.train_y_fn + '.npy') or clobber:
        feature_extraction(train_images, image_dim, prepare_data.train_x_fn, prepare_data.train_y_fn, target_dir, clobber)
    if not exists(prepare_data.validate_x_fn + '.npy') or not exists(prepare_data.validate_y_fn + '.npy') or clobber:
        feature_extraction(validate_images, image_dim, prepare_data.validate_x_fn, prepare_data.validate_y_fn, target_dir, clobber)

    prepare_data.batch_size = adjust_batch_size(train_valid_samples, validate_valid_samples, batch_size)
    if not exists(prepare_data.train_onehot_fn) or clobber: 
        set_onehot(train_images, prepare_data.train_onehot_fn)
    if not exists(prepare_data.validate_onehot_fn) or clobber: 
        set_onehot(validate_images, prepare_data.validate_onehot_fn)

    return [ images, image_dim ] 
