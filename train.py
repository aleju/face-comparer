# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import time
import datetime
import sys
import random
import os
import math
import re
import numpy as np
import copy
import csv
from collections import defaultdict
from scipy import misc
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import decomposition

from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Reshape, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adagrad, SGD
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.utils import generic_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU

from ImageAugmenter import ImageAugmenter
from MyMerge import MyMerge
from Plotter import Plotter
import matplotlib.pyplot as plt
from util import load_model, save_model_config, save_model_weights, save_optimizer_state
from skimage import transform
from MyBatchNormalization import MyBatchNormalization

SEED = 42
LFWCROP_GREY_FILEPATH = "/media/aj/grab/ml/datasets/lfwcrop_grey"
IMAGES_FILEPATH = LFWCROP_GREY_FILEPATH + "/faces"
TRAIN_COUNT_EXAMPLES = 20000
VALIDATION_COUNT_EXAMPLES = 256
TEST_COUNT_EXAMPLES = 0
EPOCHS = 1000 * 1000
BATCH_SIZE = 64
SAVE_PLOT_FILEPATH = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/plots/{identifier}.png"
SAVE_DISTRIBUTION_PLOT_FILEPATH = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/plots/{identifier}_distribution.png"
SAVE_CSV_FILEPATH = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/csv/{identifier}.csv"
SAVE_WEIGHTS_DIR = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/weights"
SAVE_OPTIMIZER_STATE_DIR = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/optimizer_state"
SAVE_CODE_DIR = "/media/aj/ssd2a/nlp/python/nn_face_comparer/experiments/code/{identifier}"
SAVE_WEIGHTS_AFTER_EPOCHS = 20
SAVE_WEIGHTS_AT_END = False
SHOW_PLOT_WINDOWS = True
Y_SAME = 1
Y_DIFFERENT = 0

np.random.seed(SEED)
random.seed(SEED)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier")
    args = parser.parse_args()
    validate_identifier(args.identifier)
    
    print("-----------------------")
    print("Loading validation dataset...")
    print("-----------------------")
    print("")
    pairs_val = get_image_pairs(IMAGES_FILEPATH, VALIDATION_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=list(), seed=SEED, verbose=True)
    
    print("-----------------------")
    print("Loading training dataset...")
    print("-----------------------")
    print("")
    pairs_train = get_image_pairs(IMAGES_FILEPATH, TRAIN_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=pairs_val, seed=SEED, verbose=True)
    print("-----------------------")
    
    assert len(pairs_val) == VALIDATION_COUNT_EXAMPLES
    assert len(pairs_train) == TRAIN_COUNT_EXAMPLES
    
    X_val, y_val = image_pairs_to_xy(pairs_val)
    X_train, y_train = image_pairs_to_xy(pairs_train)
    
    plot_person_img_distribution(
        img_filepaths_test, img_filepaths_val, img_filepaths_train,
        only_y_value=Y_SAME,
        show_plot_windows=SHOW_PLOT_WINDOWS,
        save_to_filepath=SAVE_DISTRIBUTION_PLOT_FILEPATH
    )
    
    
    print("Creating model...")
        
    model = Sequential()
    
    model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(Convolution2D(32, 32, 3, 3, border_mode='full'))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(Convolution2D(64, 64, 3, 3, border_mode='full'))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.50))
    
    model.add(Reshape(64*4, int(836/4)))
    model.add(MyBatchNormalization((64*4, int(836/4))))
    
    model.add(GRU(836/4, 64, return_sequences=True))
    model.add(Flatten())
    model.add(MyBatchNormalization((64*(64*4),)))
    model.add(Dropout(0.50))
    model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    la_plotter = LossAccPlotter(save_to_filepath=SAVE_PLOT_FILEPATH.format(identifier=args.identifier))

    epoch_start = 0
    if args.load:
        print("Loading previous model...")
        epoch_start = load_previous_model(args.load, model, optimizer, la_plotter)
    
    print("Training...")
    train_loop(args.identifier, model, optimizer, epoch_start)
    
    print("Finished.")

def validate_identifier(identifier):
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 and _ are allowed.")

def load_previous_model():
    # load old model if that is requested
    if continue_identifier:
        # load optimizer state
        (success, last_epoch) = load_optimizer_state(optimizer, cfg["save_optimizer_state_dir"], continue_identifier)
        
        # load weights
        (success, last_epoch) = load_weights(model, cfg["save_weights_dir"], continue_identifier)
        
        if not success:
            raise Exception("Cannot continue previous experiment, because no weights were saved (yet?).")
        else:
            # load previous loss/acc values per epoch from csv file
            csv_lines = open(cfg["save_csv_filepath"], "r").readlines()
            csv_lines = csv_lines[1:] # no header
            csv_cells = [line.strip().split(",") for line in csv_lines]
            epochs = [int(cells[0]) for cells in csv_cells]
            stats_train_loss = [float(cells[1]) for cells in csv_cells]
            stats_val_loss = [float(cells[2]) for cells in csv_cells]
            stats_train_acc = [float(cells[3]) for cells in csv_cells]
            stats_val_acc = [float(cells[4]) for cells in csv_cells]
            
            if last_epoch == "last":
                start_epoch = epochs[-1] + 1
            else:
                start_epoch = last_epoch + 1
                # the csv file contents may have stats logged after
                # the last checkpoint at which the weights were saved.
                # Those stats are clipped off.
                stats_train_loss = stats_train_loss[0:last_epoch+1]
                stats_val_loss = stats_val_loss[0:last_epoch+1]
                stats_train_acc = stats_train_acc[0:last_epoch+1]
                stats_val_acc = stats_val_acc[0:last_epoch+1]
    else:
        # Lists with values of loss and acc (for training and val. data)
        stats_train_loss = []
        stats_val_loss = []
        stats_train_acc = []
        stats_val_acc = []
    
        start_epoch = 0

def train_loop(args.identifier, model, optimizer, epoch_start):
    

if __name__ == "__main__":
    main()
