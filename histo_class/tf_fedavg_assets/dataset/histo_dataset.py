import codecs
import os
import sys
import zipfile
import pathlib

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


def setup_histo(data_path, N_CLIENTS):
    # Download the dataset
    # Proportion of train and test samples is arbitrary set to 80% - 20%
    (train_ds, test_ds), metadata = tfds.load('colorectal_histology',split=['train[:80%]', 'train[80%:]'],with_info=True,
        as_supervised=True)

    # separating images from labels 
    train_images_ds = train_ds.map(lambda x,y: x)
    train_labels_ds = train_ds.map(lambda x,y: y)
    test_images_ds = test_ds.map(lambda x,y: x)
    test_labels_ds = test_ds.map(lambda x,y: y)

    # converting datasets into numpy arrays of shapes (nb_img, 150, 150, 3) and (nb_labels, )
    train_images = np.array(list(train_images_ds.as_numpy_iterator()))
    train_labels = np.array(list(train_labels_ds.as_numpy_iterator()))
    test_images = np.array(list(test_images_ds.as_numpy_iterator()))
    test_labels = np.array(list(test_labels_ds.as_numpy_iterator()))

    # Split array into the number of organization
    train_images_folds = np.split(train_images, N_CLIENTS)
    train_labels_folds = np.split(train_labels, N_CLIENTS)
    test_images_folds = np.split(test_images, N_CLIENTS)
    test_labels_folds = np.split(test_labels, N_CLIENTS)

    # Save splits in different folders to simulate the different organization
    for i in range(N_CLIENTS):

        # Save train dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/train"), exist_ok=True)
        filename = data_path / f"org_{i+1}/train/train_images.npy"
        np.save(str(filename), train_images_folds[i])
        filename = data_path / f"org_{i+1}/train/train_labels.npy"
        np.save(str(filename), train_labels_folds[i])

        # Save test dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/test"), exist_ok=True)
        filename = data_path / f"org_{i+1}/test/test_images.npy"
        np.save(str(filename), test_images_folds[i])
        filename = data_path / f"org_{i+1}/test/test_labels.npy"
        np.save(str(filename), test_labels_folds[i])