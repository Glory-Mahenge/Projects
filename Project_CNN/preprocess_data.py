'''preprocess_data.py
Preprocessing data in STL-10 image dataset
OPEOLUWA SHODIPE & GLORY MAHENGE
Neural Networks
Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset
import os


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    Outline:
    1) Cast imgs to float64
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Treating the pixels as features, standardize the features "seperately"
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    imgs = imgs.astype(np.float64)
    imgs = imgs.transpose(0, 3, 1, 2) 

    means = np.mean(imgs, axis = 0)
    sd = np.std(imgs, axis = 0)

    imgs = (imgs - means) / sd
    labels = labels - 1

    return imgs, labels

def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    Outline:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)

     Resist the urge to shuffle the data here! It is best to shuffle the data "live"
    during training so hold off on shuffling your data here.
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return
    
    x_train = data[:n_train_samps]
    y_train = y[:n_train_samps]
    x_test = data[n_train_samps:n_train_samps + n_test_samps]
    y_test = y[n_train_samps:n_train_samps + n_test_samps]
    x_val = data[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]
    y_val = y[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]
    x_dev = data[n_train_samps + n_test_samps + n_valid_samps:n_train_samps + n_test_samps + n_valid_samps + n_dev_samps]
    y_dev = y[n_train_samps + n_test_samps + n_valid_samps :n_train_samps + n_test_samps + n_valid_samps + n_dev_samps]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev



def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500,scale_fact=None):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    stl_imgs, stl_labels = load_stl10_dataset.load(scale_fact=scale_fact)
    stl_imgs_pp, stl_labels_pp = preprocess_stl(stl_imgs, stl_labels)
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = create_splits(stl_imgs_pp, stl_labels_pp, n_train_samps=n_train_samps, n_test_samps=n_test_samps, n_valid_samps=n_valid_samps, n_dev_samps=n_dev_samps)  
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev
