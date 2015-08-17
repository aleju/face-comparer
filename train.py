# -*- coding: utf-8 -*-
"""The main training file.
Traings a neural net to compare pairs of faces with each other.

Call this file in the following way:

    python train.py name_of_experiment

or more complex:

    python train.py name_of_experiment --load="old_experiment_name" --dropout=0.0 --augmul=1.0

where
    name_of_experiment:
        Is the name of this experiment, used when saving data, e.g. "exp5_more_dropout".
    --load="old_experiment_name":
        Is the name of an old experiment to continue. Must have the identical
        network architecture and optimizer as the new network.
    --dropout=0.0:
        Dropout strength to use for the last two dropout layers.
    --augmul=1.0:
        Augmentation strength to use when augmentating images (e.g. rotation, shift).
        0.5 is weak, 1.0 is normal, 1.5+ is strong.
"""
from __future__ import absolute_import, division, print_function

import sys
import random
import os
import re
import numpy as np
import csv
import argparse
import math
#import matplotlib.pyplot as plt

from scipy import misc # for resizing of images

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Activation, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adagrad, Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU, LSTM
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.utils import generic_utils

from ImageAugmenter import ImageAugmenter
from laplotter import LossAccPlotter
from utils import save_model_weights, save_optimizer_state, load_weights, load_optimizer_state
from datasets import get_image_pairs, image_pairs_to_xy

SEED = 42
LFWCROP_GREY_FILEPATH = "/media/aj/grab/ml/datasets/lfwcrop_grey"
IMAGES_FILEPATH = LFWCROP_GREY_FILEPATH + "/faces"
TRAIN_COUNT_EXAMPLES = 20000
VALIDATION_COUNT_EXAMPLES = 256
TEST_COUNT_EXAMPLES = 0
EPOCHS = 1000 * 1000
BATCH_SIZE = 64
SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/experiments"
SAVE_PLOT_FILEPATH = "%s/plots/{identifier}.png" % (SAVE_DIR)
SAVE_DISTRIBUTION_PLOT_FILEPATH = "%s/plots/{identifier}_distribution.png" % (SAVE_DIR)
SAVE_CSV_FILEPATH = "%s/csv/{identifier}.csv" % (SAVE_DIR)
SAVE_WEIGHTS_DIR = "%s/weights" % (SAVE_DIR)
SAVE_OPTIMIZER_STATE_DIR = "%s/optstate" % (SAVE_DIR)
SAVE_CODE_DIR = "%s/code/{identifier}" % (SAVE_DIR)
SAVE_WEIGHTS_AFTER_EPOCHS = 1
SHOW_PLOT_WINDOWS = True
Y_SAME = 1
Y_DIFFERENT = 0

np.random.seed(SEED)
random.seed(SEED)

def main():
    """Main function.
    1. Handle console arguments,
    2. Load datasets,
    3. Initialize network,
    4. Initialize training looper
    5. Train (+validate)."""
    
    # handle arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier", help="A short name/identifier for your experiment, e.g. 'ex42b_more_dropout'.")
    parser.add_argument("--load", required=False, help="Identifier of a previous experiment that you want to continue (loads weights, optimizer state and history).")
    parser.add_argument("--dropout", required=False, help="Dropout rate (0.0 - 1.0) after the last conv-layer and after the GRU layer. Default is 0.0.")
    parser.add_argument("--augmul", required=False, help="Multiplicator for the augmentation (0.0=no augmentation, 1.0=normal aug., 2.0=rather strong aug.). Default is 1.0.")
    args = parser.parse_args()
    validate_identifier(args.identifier, must_exist=False)
    if args.load:
        validate_identifier(args.load)

    if identifier_exists(args.identifier):
        if args.identifier != args.load:
            agreed = ask_continue("[WARNING] Identifier '%s' already exists and is different from load-identifier '%s'. It will be overwritten. Continue? [y/n]" % (args.identifier, args.load))
            if not agreed:
                return

    if args.augmul is None:
        args.augmul = 1.0

    # load validation set
    # we load this before the training set so that it is less skewed (otherwise
    # most images of people with only one image would be lost to the training set)
    print("-----------------------")
    print("Loading validation dataset...")
    print("-----------------------")
    print("")
    pairs_val = get_image_pairs(IMAGES_FILEPATH, VALIDATION_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=list(), seed=SEED, verbose=True)

    # load training set
    print("-----------------------")
    print("Loading training dataset...")
    print("-----------------------")
    print("")
    pairs_train = get_image_pairs(IMAGES_FILEPATH, TRAIN_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=pairs_val, seed=SEED, verbose=True)
    print("-----------------------")

    # check if more pairs have been requested than can be generated
    assert len(pairs_val) == VALIDATION_COUNT_EXAMPLES
    assert len(pairs_train) == TRAIN_COUNT_EXAMPLES

    # we loaded pairs of filepaths so far, now load the contents
    print("Loading image contents from hard drive...")
    X_val, y_val = image_pairs_to_xy(pairs_val)
    X_train, y_train = image_pairs_to_xy(pairs_train)

    """
    plot_person_img_distribution(
        img_filepaths_test, img_filepaths_val, img_filepaths_train,
        only_y_value=Y_SAME,
        show_plot_windows=SHOW_PLOT_WINDOWS,
        save_to_filepath=SAVE_DISTRIBUTION_PLOT_FILEPATH
    )
    """

    # initialize the network
    print("Creating model...")
    model, optimizer = create_model(args.dropout)
    #model, optimizer = create_model1b(args.dropout)
    #model, optimizer = create_model2(args.dropout)
    #model, optimizer = create_model3(args.dropout)
    #model, optimizer = create_model4(args.dropout)
    #model, optimizer = create_model_nonorm(args.dropout)
    #model, optimizer = create_model_full_border(args.dropout)
    #model, optimizer = create_model_td_dense(args.dropout)
    
    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    # -------------------
    # Training loop part
    # -------------------
    # initialize the plotter for loss and accuracy
    la_plotter = LossAccPlotter(save_to_filepath=SAVE_PLOT_FILEPATH.format(identifier=args.identifier))
    """
    ia_train = ImageAugmenter(64, 64, hflip=True, vflip=False,
                              scale_to_percent=1.15, scale_axis_equally=False,
                              rotation_deg=25, shear_deg=8,
                              translation_x_px=7, translation_y_px=7)
    """
    # intialize the image augmenters
    # they are going to rotate, shift etc. the images
    augmul = float(args.augmul)
    ia_train = ImageAugmenter(64, 64, hflip=True, vflip=False,
                              scale_to_percent=1.0 + (0.075*augmul),
                              scale_axis_equally=False,
                              rotation_deg=int(7*augmul),
                              shear_deg=int(3*augmul),
                              translation_x_px=int(3*augmul),
                              translation_y_px=int(3*augmul))
    # prefill the training augmenter with lots of random affine transformation
    # matrices, so that they can be reused many times
    ia_train.pregenerate_matrices(15000)
    
    # we dont want any augmentations for the validation set
    ia_val = ImageAugmenter(64, 64)

    # load previous data if requested
    # includes: weights (works only if new and old model are identical),
    # optimizer state (works only for same optimizer, seems to cause errors for adam),
    # history (loss and acc values per epoch),
    # old plot (will be continued)
    if args.load:
        print("Loading previous model...")
        epoch_start, history = load_previous_model(args.load, model, optimizer, la_plotter)
    else:
        epoch_start = 0
        history = History()
    
    # run the training loop
    print("Training...")
    train_loop(args.identifier, model, optimizer, epoch_start, history, la_plotter, ia_train, ia_val, X_train, y_train, X_val, y_val)
    
    print("Finished.")

def validate_identifier(identifier, must_exist=True):
    """Check whether a used identifier is a valid one or raise an error.
    
    Optionally also check if there is already an experiment with the identifier
    and raise an error if there is none yet.
    
    Valid identifiers contain only:
        a-z
        A-Z
        0-9
        _
    
    Args:
        identifier: Identifier to check for validity.
        must_exist: If set to true and no experiment uses the identifier yet,
            an error will be raised.
    
    Returns:
        void
    """
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 and _ are allowed.")
    if must_exist:
        if not identifier_exists(identifier):
            raise Exception("No model with identifier '{}' seems to exist.".format(identifier))

def identifier_exists(identifier):
    """Returns True if the provided identifier exists.
    The existence and check by checking if there is a history (csv file)
    with the provided identifier.
    
    Args:
        identifier: Identifier of the experiment.

    Returns:
        True if an experiment with the identifier exists.
        False otherwise.
    """
    filepath = SAVE_CSV_FILEPATH.format(identifier=identifier)
    if os.path.isfile(filepath):
        return True
    else:
        return False

def ask_continue(message):
    """Displays the message and waits for a "y" (yes) or "n" (no) input by the user.
    
    Args:
        message: The message to display.

    Returns:
        True if the user has entered "y" (for yes).
        False if the user has entered "n" (for no).
    """
    choice = raw_input(message)
    while choice not in ["y", "n"]:
        choice = raw_input("Enter 'y' (yes) or 'n' (no) to continue.")
    return choice == "y"

def load_previous_model(identifier, model, optimizer, la_plotter):
    """Loads a previous model with the provided identifier (weights, optimizer state,
    history, plot).
    
    Args:
        identifier: Identifier of the previous experiment.
        model: The current model. That model's weights will be changed to the loaded ones.
            Architecture (layers) must be identical.
        optimizer: The current optimizer. That optimizer's state will be changed to
            the loaded one.
        la_plotter: The current plotter for loss and accuracy. Will be updated
            with the loaded history data.
    
    Returns:
        Will return a tupel (last epoch, history), where "last epoch" is the
        last epoch that was finished in the old experiment and "history"
        is the old experiment's history object (i.e. epochs, loss, acc).
    """
    # load optimizer state
    (success, last_epoch) = load_optimizer_state(optimizer, SAVE_OPTIMIZER_STATE_DIR, identifier)
    if not success:
        print("[WARNING] could not successfully load optimizer state of identifier '{}'.".format(identifier))
    
    # load weights
    # we overwrite the results of the optimizer loading here, because errors
    # there are not very important, we can still go on training.
    (success, last_epoch) = load_weights(model, SAVE_WEIGHTS_DIR, identifier)
    
    if not success:
        raise Exception("Cannot continue previous experiment, because no weights were saved (yet?).")
    
    # load history from csv file
    history = History()
    #history.load_from_file("{}/{}.csv".format(SAVE_CSV_DIR, identifier))
    history.load_from_file(SAVE_CSV_FILEPATH.format(identifier=identifier), last_epoch=last_epoch)
    #history = load_history(SAVE_CSV_DIR, identifier, last_epoch=last_epoch)
    
    # update loss acc plotter
    for i, epoch in enumerate(history.epochs):
        la_plotter.add_values(epoch,
                              loss_train=history.loss_train[i], loss_val=history.loss_val[i],
                              acc_train=history.acc_train[i], acc_val=history.acc_val[i],
                              redraw=False)
    
    return history.epochs[-1], history

"""
def load_history(save_history_dir, identifier):
    # load previous loss/acc values per epoch from csv file
    csv_filepath = "{}/{}.csv".format(save_history_dir, identifier)
    csv_lines = open(csv_filepath, "r").readlines()
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
    
    epochs = range(start_epoch)
    history.add_all(start_epoch,
                    stats_train_loss[0:start_epoch],
                    stats_train_val[0:start_epoch],
                    stats_acc_train[0:start_epoch],
                    stats_acc_val[0:start_epoch])
    return history
"""

def create_model(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 32 x 32+2 x 64+2 = 32x34x66
    model.add(Convolution2D(32, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 32 x 34-2 x 66-2 = 32x32x64
    model.add(Convolution2D(32, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32 x 32/2 x 64/2 = 32x16x32
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64 x 16-2 x 32-2 = 64x14x30
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 14-2 x 30-2 = 64x12x28
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(dropout))
    
    # 64x12x28 = 64x336 = 21504
    # In 64*4 slices: 64*4 x 336/4 = 256x84
    model.add(Reshape(64*4, int(336/4)))
    model.add(BatchNormalization((64*4, int(336/4))))
    
    model.add(GRU(336/4, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((64*(64*4),)))
    model.add(Dropout(dropout))
    
    model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model1b(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 32 x 32+2 x 64+2 = 32x34x66
    model.add(Convolution2D(32, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 32 x 34-2 x 66-2 = 32x32x64
    model.add(Convolution2D(32, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32 x 32/2 x 64/2 = 32x16x32
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64 x 16-2 x 32-2 = 64x14x30
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 14-2 x 30-2 = 64x12x28
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(dropout))
    
    # 64x12x28 = 64x336 = 21504
    # In 64*4 slices: 64*4 x 336/4 = 256x84
    model.add(Reshape(64*4, int(336/4)))
    model.add(BatchNormalization((64*4, int(336/4))))
    
    model.add(GRU(336/4, 128, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((128*(64*4),)))
    model.add(Dropout(dropout))
    
    model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model4(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 64 x 32+2 x 64+2 = 64x34x66
    model.add(Convolution2D(32, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 34-2 x 66-2 = 64x32x64
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 32-2 x 64-2 = 64x30x62
    model.add(Convolution2D(128, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 128x15x31
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 128x15x31 = 128x465 = 2*29760
    model.add(Reshape(128, 465))
    model.add(BatchNormalization((128, 465)))
    
    """
    model.add(TimeDistributedDense(465, 128, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization((64, 128)))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    """
    
    model.add(LSTM(465, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((128*64,)))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.10))
    model.add(GaussianDropout(0.10))
    
    model.add(Dense(128*64, 128, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization((128,)))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    model.add(Dense(128, 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model2(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    model.add(GaussianNoise(0.05))
    
    # 4x34x66
    model.add(Convolution2D(4, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 8x32x64
    model.add(Convolution2D(8, 4, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 16x30x62
    model.add(Convolution2D(16, 8, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32x28x60
    model.add(Convolution2D(32, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32x14x30
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64x12x28
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 64*12*28 = 64*336 = 64*4 * 84
    model.add(Reshape(64*4, 336//4))
    model.add(BatchNormalization((64*4, 336//4)))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    
    # GRU over 64*4 slices
    model.add(GRU(336//4, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((64*(64*4),)))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    
    # Dense from 64*4 slices, each 64 nodes (64*4*64) into 64 nodes
    model.add(Dense(64*4*64, 64, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization((64,)))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.05))
    model.add(GaussianDropout(0.05))
    
    # Output
    model.add(Dense(64, 1, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model3(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    model.add(GaussianNoise(0.05))
    
    # 8x34x66
    model.add(Convolution2D(8, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    # 16x32x64
    model.add(Convolution2D(16, 8, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    # 16x16x32
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 32x14x30
    model.add(Convolution2D(32, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    # 64x12x28
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    # 64x6x14
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 128x4x12
    model.add(Convolution2D(128, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    model.add(GaussianNoise(0.00))
    model.add(GaussianDropout(0.00))
    
    # 128*4*12 = 128*48
    model.add(Reshape(128, 48))
    model.add(BatchNormalization((128, 48)))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    
    # LSTM over 128 slices
    model.add(LSTM(48, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((128*64),))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    
    # Dense from 64*4 slices, each 64 nodes (64*4*64) into 64 nodes
    model.add(Dense(128*64, 64, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(LeakyReLU(0.33))
    model.add(BatchNormalization((64,)))
    model.add(Dropout(dropout))
    model.add(GaussianNoise(0.05))
    model.add(GaussianDropout(0.05))
    
    # Output
    model.add(Dense(64, 1, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model_nonorm(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 32 x 32+2 x 64+2 = 32x34x66
    model.add(Convolution2D(32, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 32 x 34-2 x 66-2 = 32x32x64
    model.add(Convolution2D(32, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32 x 32/2 x 64/2 = 32x16x32
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64 x 16-2 x 32-2 = 64x14x30
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 14-2 x 30-2 = 64x12x28
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(dropout))
    
    # 64x12x28 = 64x336 = 21504
    # In 64*4 slices: 64*4 x 336/4 = 256x84
    model.add(Reshape(64*4, int(336/4)))
    #model.add(BatchNormalization((64*4, int(336/4))))
    model.add(GaussianNoise(0.1))
    
    model.add(GRU(336/4, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((64*(64*4),)))
    model.add(GaussianNoise(0.1))
    model.add(Dropout(dropout))
    
    model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model_full_border(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 32 x 32+2 x 64+2 = 32x34x66
    model.add(Convolution2D(32, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 32 x 34+2 x 66+2 = 32x36x68
    model.add(Convolution2D(32, 32, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 32 x 36/2 x 68/2 = 32x18x34
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64 x 18+2 x 34+2 = 64x20x36
    model.add(Convolution2D(64, 32, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 20+2 x 36+2 = 64x22x38
    model.add(Convolution2D(64, 64, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(dropout))
    
    # 64x22x38 = 64x836 = 53504
    # In 64*4 slices: 64*4 x 836/4 = 256x209
    model.add(Reshape(64*4, int(836/4)))
    model.add(BatchNormalization((64*4, int(836/4))))
    
    model.add(GRU(836/4, 64, return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization((64*(64*4),)))
    model.add(Dropout(dropout))
    
    model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    optimizer = Adagrad()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

def create_model_td_dense(dropout=None):
    dropout = float(dropout) if dropout is not None else 0.00
    print("Dropout will be set to {}".format(dropout))
    
    model = Sequential()
    
    # 8 x 32+2 x 64+2 = 8x34x66
    model.add(Convolution2D(8, 1, 3, 3, border_mode="full"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 16 x 34-2 x 66-2 = 16x32x64
    model.add(Convolution2D(16, 8, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    
    # 16 x 32/2 x 64/2 = 16x16x32
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 32 x 16-2 x 32-2 = 32x14x30
    model.add(Convolution2D(32, 16, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))
    # 64 x 14-2 x 30-2 = 64x12x28
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    
    # 64 x 12/2 x 28/2 = 64x6x14
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # 64x6x14 = 64x84 = 5376
    # In 64*4 slices: 64*4 x 84/4 = 256x21
    model.add(Reshape(64*4, 84//4))
    model.add(BatchNormalization((64*4, 84//4)))
    model.add(Dropout(dropout))
    
    model.add(TimeDistributedDense(84//4, 1, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(Activation("sigmoid"))
    model.add(Flatten())
    model.add(Dropout(0.00))
    
    model.add(Dense(256, 1, init="glorot_uniform", W_regularizer=l2(0.00001)))
    model.add(Activation("sigmoid"))

    #optimizer = Adagrad()
    optimizer = Adam()
    
    print("Compiling model...")
    model.compile(loss="binary_crossentropy", class_mode="binary", optimizer=optimizer)
    
    return model, optimizer

class History(object):
    """A simple object to save the training history of an experiment to.
    History included: Epochs, Loss (training set), Loss (validation set),
    Accuracy (training set), Accuracy (validation set).
    
    Can easily be saved to a csv file.
    """
    def __init__(self):
        """Initialize the history."""
        #self.first_epoch = 1000 * 1000
        #self.last_epoch = -1
        self.epochs = []
        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []

    def add(self, epoch, loss_train=None, loss_val=None, acc_train=None, acc_val=None):
        """Add an entry (row) to the history.
        
        Should work in principle without providing all values, but nevertheless
        you should provide all. Used named attributes here mostly for clarity when
        calling the function, so that values don't get mixed up.
        
        Args:
            epoch: The epoch of the other values (i.e. of the row).
            loss_train: The loss value of the training set of the epoch.
            loss_val: The loss value of the validation set of the epoch.
            acc_train: The accuracy value of the training set of the epoch.
            acc_val: The accuracy value of the validation set of the epoch.
        """
        self.epochs.append(epoch)
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)
        self.acc_train.append(acc_train)
        self.acc_val.append(acc_val)
        #self.first_epoch = min(self.first_epoch, epoch)
        #self.last_epoch = max(self.last_epoch, epoch)

    def add_all(self, start_epoch, loss_train, loss_val, acc_train, acc_val):
        """Add lists of values to the history.
        
        All lists must have equal lengths.
        
        Args:
            start_epoch: Epoch of the first value.
            loss_train: List of the values of the loss of the training set.
            loss_val: List of the values of the loss of the validation set.
            acc_train: List of the values of the accuracy of the training set.
            acc_val: List of the values of the accuracy of the validation set.
        """
        last_epoch = start_epoch + len(loss_train)
        for epoch, lt, lv, at, av in zip(range(start_epoch, last_epoch+1), loss_train, loss_val, acc_train, acc_val):
            self.add(epoch, loss_train=lt, loss_val=lv, acc_train=at, acc_val=av)

    def save_to_filepath(self, csv_filepath):
        """Saves the contents of the history to a csv file.
        
        Args:
            csv_filepath: Full path to the file to write to. All content in the
                file will be completely overwritten.
        """
        with open(csv_filepath, "w") as fp:
            csvw = csv.writer(fp, delimiter=",")
            # header row
            rows = [["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]]
            
            #data = data + [[r_e, r_tl, r_vl, r_ta, r_va] for r_e, r_tl, r_vl, r_ta, r_va in zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc)]
            rows.extend(zip(self.epochs, self.loss_train, self.loss_val, self.acc_train, self.acc_val))
            csvw.writerows(rows)

    def load_from_file(self, csv_filepath, last_epoch=None):
        """Loads the content of the history from a csv file.
        
        It is assumed that the csv file has the same structure as the one
        created by save_to_filepath().
        
        Args:
            csv_filepath: Full path to the file to read.
            last_epoch: The epoch until which to read the content (including).
                E.g. last_epoch=10 will read the rows for epoch 1, 2, 3, ... and 10.
                If set to "last" or None then all epochs will be read.
                Default is None (read all).
        """
        # load previous loss/acc values per epoch from csv file
        csv_lines = open(csv_filepath, "r").readlines()
        csv_lines = csv_lines[1:] # no header
        csv_cells = [line.strip().split(",") for line in csv_lines]
        epochs = [int(cells[0]) for cells in csv_cells]
        stats_loss_train = [float(cells[1]) for cells in csv_cells]
        stats_loss_val = [float(cells[2]) for cells in csv_cells]
        stats_acc_train = [float(cells[3]) for cells in csv_cells]
        stats_acc_val = [float(cells[4]) for cells in csv_cells]
        
        if last_epoch is not None and last_epoch is not "last":
            epochs = epochs[0:last_epoch+1]
            stats_loss_train = stats_loss_train[0:last_epoch+1]
            stats_loss_val = stats_loss_val[0:last_epoch+1]
            stats_acc_train = stats_acc_train[0:last_epoch+1]
            stats_acc_val = stats_acc_val[0:last_epoch+1]
        
        self.epochs = epochs
        self.loss_train = stats_loss_train
        self.loss_val = stats_loss_val
        self.acc_train = stats_acc_train
        self.acc_val = stats_acc_val

def train_loop(identifier, model, optimizer, epoch_start, history, la_plotter, ia_train, ia_val, X_train, y_train, X_val, y_val):
    """Perform the training loop.
    
    Args:
        identifier: Identifier of the experiment.
        model: The network to train (and validate).
        optimizer: The network's optimizer (e.g. SGD).
        epoch_start: The epoch to start the training at. Usually 0, but
            can be higher if an old training is continued/loaded.
        history: The history for this training.
            Can be filled already, if an old training is continued/loaded.
        la_plotter: The plotter used to plot loss and accuracy values.
            Can be filled already, if an old training is continued/loaded.
        ia_train: ImageAugmenter to use to augment the training images.
        ia_val: ImageAugmenter to use to augment the validation images.
        X_train: The training set images.
        y_train: The training set labels (same persons, different persons).
        X_val: The validation set images.
        y_val: The validation set labels.
    """
    
    # Loop over each epoch, i.e. executes 20 times if epochs set to 20
    # start_epoch is not 0 if we continue an older model.
    for epoch in range(epoch_start, EPOCHS):
        print("Epoch", epoch)
        
        # Variables to collect the sums for loss and accuracy (for training and
        # validation dataset). We will use them to calculate the loss/acc per
        # example (which will be ploted and added to the history).
        loss_train_sum = 0
        loss_val_sum = 0
        acc_train_sum = 0
        acc_val_sum = 0
        
        nb_examples_train = X_train.shape[0]
        nb_examples_val = X_val.shape[0]
        
        # Training loop
        progbar = generic_utils.Progbar(nb_examples_train)
        
        for X_batch, Y_batch in flow_batches(X_train, y_train, ia_train, shuffle=True, train=True):
            loss, acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(len(X_batch), values=[("train loss", loss), ("train acc", acc)])
            loss_train_sum += (loss * len(X_batch))
            acc_train_sum += (acc * len(X_batch))
        
        # Validation loop
        progbar = generic_utils.Progbar(nb_examples_val)
        
        # Iterate over each batch in the validation data
        # and calculate loss and accuracy for each batch
        for X_batch, Y_batch in flow_batches(X_val, y_val, ia_val, shuffle=False, train=False):
            loss, acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(len(X_batch), values=[("val loss", loss), ("val acc", acc)])
            loss_val_sum += (loss * len(X_batch))
            acc_val_sum += (acc * len(X_batch))

        # Calculate the loss and accuracy for this epoch
        # (averaged over all training data batches)
        loss_train = loss_train_sum / nb_examples_train
        acc_train = acc_train_sum / nb_examples_train
        loss_val = loss_val_sum / nb_examples_val
        acc_val = acc_val_sum / nb_examples_val
        
        history.add(epoch, loss_train=loss_train, loss_val=loss_val, acc_train=acc_train, acc_val=acc_val)
        
        # Update plots with new data from this epoch
        # We start plotting _after_ the first epoch as the first one usually contains
        # a huge fall in loss (increase in accuracy) making it harder to see the
        # minor swings at epoch 1000 and later.
        if epoch > 0:
            la_plotter.add_values(epoch, loss_train=loss_train, loss_val=loss_val, acc_train=acc_train, acc_val=acc_val)
        
        # Save the history to a csv file
        if SAVE_CSV_FILEPATH is not None:
            csv_filepath = SAVE_CSV_FILEPATH.format(identifier=identifier)
            history.save_to_filepath(csv_filepath)
        
        # Save the weights and optimizer state to files
        swae = SAVE_WEIGHTS_AFTER_EPOCHS
        if swae and swae > 0 and (epoch+1) % swae == 0:
            print("Saving model...")
            #save_model_weights(model, cfg["save_weights_dir"], model_name + ".at" + str(epoch) + ".weights")
            #save_optimizer_state(optimizer, cfg["save_optimizer_state_dir"], model_name + ".at" + str(epoch) + ".optstate", overwrite=True)
            save_model_weights(model, SAVE_WEIGHTS_DIR, "{}.last.weights".format(identifier), overwrite=True)
            save_optimizer_state(optimizer, SAVE_OPTIMIZER_STATE_DIR, "{}.last.optstate".format(identifier), overwrite=True)

def flow_batches(X_in, y_in, ia, batch_size=BATCH_SIZE, shuffle=False, train=False):
    """Uses the datasets (either train. or val.) and returns them batch by batch,
    transformed via provided ImageAugmenter (ia).
    
    Args:
        X_in: Pairs of input images of shape (N, 2, 64, 64).
        y_in: Labels for the pairs of shape (N, 1).
        ia: ImageAugmenter to use.
        batch_size: Size of the batches to return.
        shuffle: Whether to shuffle the order of the images before starting to
            return any batches.

    Returns:
        Batches, i.e. tuples of (X, y).
    """
    
    # Shuffle the datasets before starting to return batches
    if shuffle:
        # we copy X_in and y_in here, otherwise the original X_in and y_in
        # will be shuffled by numpy too.
        X = np.copy(X_in)
        y = np.copy(y_in)

        state = np.random.get_state()
        seed = random.randint(1, 10e6)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.set_state(state)
    else:
        X = X_in
        y = y_in
    
    # Iterate over every possible batch and collect the examples
    # for that batch
    nb_examples = X.shape[0]
    nb_batch = int(math.ceil(float(nb_examples)/batch_size))
    for b in range(nb_batch):
        batch_end = (b+1)*batch_size
        if batch_end > nb_examples:
            nb_samples = nb_examples - b*batch_size
        else:
            nb_samples = batch_size
        
        # extract all images of the batch from X
        batch_start_idx = b*batch_size
        batch = X[batch_start_idx:batch_start_idx + nb_samples]
        
        # augment the images of the batch
        batch_img1 = batch[:, 0, ...] # left images
        batch_img2 = batch[:, 1, ...] # right images
        batch_img1 = ia.augment_batch(batch_img1)
        batch_img2 = ia.augment_batch(batch_img2)
        
        # resize and merge the pairs of images to shape (B, 1, 32, 64), where
        # B is the size of this batch and 1 represents the only channel
        # of the image.
        X_batch = np.zeros((nb_samples, 1, 32, 64))
        for i in range(nb_samples):
            # sometimes switch positions (left/right) of images during training
            if train and random.random() < 0.5:
                img1 = batch_img2[i]
                img2 = batch_img1[i]
            else:
                img1 = batch_img1[i]
                img2 = batch_img2[i]

            # downsize images
            # note: imresize projects the image into 0-255, even if it was 0-1.0 before
            img1 = misc.imresize(img1, (32, 32)) / 255.0
            img2 = misc.imresize(img2, (32, 32)) / 255.0
            
            # merge the two images to one image
            X_batch[i] = np.concatenate((img1, img2), axis=1)
        
        # Collect the y values of the batch
        y_batch = y[batch_start_idx:batch_start_idx + nb_samples]

        yield X_batch, y_batch

if __name__ == "__main__":
    main()
