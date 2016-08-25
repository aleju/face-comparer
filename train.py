# -*- coding: utf-8 -*-
"""The main training file.
Traings a neural net to compare pairs of faces with each other.

Call this file in the following way:

    python train.py name_of_experiment --images="/path/to/lfwcrop_grey/faces"

or more complex:

    python train.py name_of_experiment
           --images="/path/to/lfwcrop_grey/faces"
           --load="old_experiment_name"

where
    name_of_experiment:
        Is the name of this experiment, used when saving data, e.g. "exp5_more_dropout".
    --load="old_experiment_name":
        Is the name of an old experiment to continue. Must have the identical
        network architecture and optimizer as the new network.
"""
from __future__ import absolute_import, division, print_function

import random
import os
import re
import numpy as np
import argparse
import math

from scipy import misc

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adagrad, Adam, Adamax, SGD
from keras.regularizers import l1, l2, l1l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU
from keras.layers import merge, Input
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.utils import generic_utils

from libs.ImageAugmenter import ImageAugmenter
from libs.laplotter import LossAccPlotter
from utils.saveload import load_previous_model, save_model_weights
from utils.datasets import get_image_pairs, image_pairs_to_xy, plot_dataset_skew
from utils.History import History
from utils.Progbar import Progbar

SEED = 42
TRAIN_COUNT_EXAMPLES = 20000
VALIDATION_COUNT_EXAMPLES = 256
EPOCHS = 1000 * 1000
BATCH_SIZE = 128
BATCH_SIZE_VAL = 64
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_CHANNELS = 1
SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/experiments"
SAVE_PLOT_FILEPATH = "%s/plots/{identifier}.png" % (SAVE_DIR)
SAVE_DISTRIBUTION_PLOT_FILEPATH = "%s/plots/{identifier}_dataset_skew.png" % (SAVE_DIR)
SAVE_CSV_FILEPATH = "%s/csv/{identifier}.csv" % (SAVE_DIR)
SAVE_WEIGHTS_DIR = "%s/weights" % (SAVE_DIR)
#SAVE_OPTIMIZER_STATE_DIR = "%s/optstate" % (SAVE_DIR)
SAVE_WEIGHTS_AFTER_EPOCHS = 1
SHOW_PLOT_WINDOWS = False

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
    parser.add_argument("identifier",
                        help="A short name/identifier for your experiment, " \
                             "e.g. 'ex42b_more_dropout'.")
    parser.add_argument("--images", required=True,
                        help="Filepath to the 'faces/' subdirectory in the " \
                             "'Labeled Faces in the Wild grayscaled and " \
                             "cropped' dataset.")
    parser.add_argument("--load", required=False,
                        help="Identifier of a previous experiment that you " \
                             "want to continue (loads weights, optimizer state "
                             "and history).")
    args = parser.parse_args()
    validate_identifier(args.identifier, must_exist=False)

    if not os.path.isdir(args.images):
        raise Exception("The provided filepath to the dataset seems to not exist.")

    if not args.images.endswith("/faces"):
        print("[WARNING] Filepath to the dataset is expected to usually end in '/faces', i.e. the default directory containing all face images in the lfwcrop_grey dataset.")

    if args.load:
        validate_identifier(args.load)

    if identifier_exists(args.identifier):
        if args.identifier != args.load:
            agreed = ask_continue("[WARNING] Identifier '%s' already exists and " \
                                  "is different from load-identifier '%s'. It " \
                                  "will be overwritten. Continue? [y/n] " \
                                  % (args.identifier, args.load))
            if not agreed:
                return

    # load validation set
    # we load this before the training set so that it is less skewed (otherwise
    # most images of people with only one image would be lost to the training set)
    print("-----------------------")
    print("Loading validation dataset...")
    print("-----------------------")
    print("")
    pairs_val = get_image_pairs(args.images, VALIDATION_COUNT_EXAMPLES,
                                pairs_of_same_imgs=False, ignore_order=True,
                                exclude_images=list(), seed=SEED, verbose=True)

    # load training set
    print("-----------------------")
    print("Loading training dataset...")
    print("-----------------------")
    print("")
    pairs_train = get_image_pairs(args.images, TRAIN_COUNT_EXAMPLES,
                                  pairs_of_same_imgs=False, ignore_order=True,
                                  exclude_images=pairs_val, seed=SEED, verbose=True)
    print("-----------------------")

    # check if more pairs have been requested than can be generated
    assert len(pairs_val) == VALIDATION_COUNT_EXAMPLES
    assert len(pairs_train) == TRAIN_COUNT_EXAMPLES

    # we loaded pairs of filepaths so far, now load the contents
    print("Loading image contents from hard drive...")
    X_val, y_val = image_pairs_to_xy(pairs_val, height=INPUT_HEIGHT, width=INPUT_WIDTH)
    X_train, y_train = image_pairs_to_xy(pairs_train, height=INPUT_HEIGHT, width=INPUT_WIDTH)

    # Plot dataset skew
    print("Saving dataset skew plot to file...")
    plot_dataset_skew(
        pairs_train, pairs_val, [],
        only_y_same=True,
        show_plot_windows=SHOW_PLOT_WINDOWS,
        save_to_filepath=SAVE_DISTRIBUTION_PLOT_FILEPATH.format(identifier=args.identifier)
    )

    # initialize the network
    print("Creating model...")
    model, optimizer = create_model()

    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    # -------------------
    # Training loop part
    # -------------------
    # initialize the plotter for loss and accuracy
    sp_fpath = SAVE_PLOT_FILEPATH.format(identifier=args.identifier)
    la_plotter = LossAccPlotter(save_to_filepath=sp_fpath, show_plot_window=SHOW_PLOT_WINDOWS)

    # initialize the image augmenter for training images
    ia_train = ImageAugmenter(INPUT_WIDTH, INPUT_HEIGHT, hflip=True, vflip=False,
                              scale_to_percent=1.1,
                              scale_axis_equally=False,
                              rotation_deg=20,
                              shear_deg=6,
                              translation_x_px=4,
                              translation_y_px=4)

    # prefill the training augmenter with lots of random affine transformation
    # matrices, so that they can be reused many times
    ia_train.pregenerate_matrices(15000)

    # we dont want any augmentations for the validation set
    ia_val = ImageAugmenter(INPUT_WIDTH, INPUT_HEIGHT)

    # load previous data if requested
    # includes: weights (works only if new and old model are identical),
    # optimizer state (works only for same optimizer, seems to cause errors for adam),
    # history (loss and acc values per epoch),
    # old plot (will be continued)
    if args.load:
        print("Loading previous model...")
        epoch_start, history = \
            load_previous_model(args.load, model, la_plotter,
                                SAVE_WEIGHTS_DIR, SAVE_CSV_FILEPATH)
    else:
        epoch_start = 0
        history = History()

    print("Model summary:")
    model.summary()

    # run the training loop
    print("Training...")
    train_loop(args.identifier, model, optimizer, epoch_start, history,
               la_plotter, ia_train, ia_val, X_train, y_train, X_val, y_val)

    print("Finished.")

def create_model(dropout=None):
    """Expects batches from flow_batches_branched()
    """


    init_conv = "orthogonal"
    init_dense = "glorot_normal"

    def conv(x, n_filters, kH, kW, sH, sW, border_same=True, drop=0.1):
        border_mode = "same" if border_same else "valid"
        x = Convolution2D(n_filters, kH, kW, subsample=(sH, sW), border_mode=border_mode, init=init_conv)(x)
        x = LeakyReLU(0.33)(x)
        if drop > 0:
            x = Dropout(drop)(x)
        return x

    face_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32")

    face = conv(face_input, 32, 3, 3, 1, 1, False) # 30x30
    face = conv(face, 32, 3, 3, 1, 1, False) # 28x28
    face = conv(face, 64, 5, 5, 2, 2, True) # 14x14
    face = conv(face, 64, 3, 3, 1, 1, False) # 12x12
    face = conv(face, 128, 5, 5, 2, 2, True) # 6x6
    face = conv(face, 128, 3, 3, 1, 1, False, drop=0.25) # 4x4
    face = Flatten()(face)
    face = Dense(512, init=init_dense)(face)
    face = BatchNormalization(mode=2)(face)
    face = Activation("tanh")(face)
    face_output = face

    face_model = Model(face_input, face_output)
    face_left_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32", name="face_left")
    face_right_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32", name="face_right")
    face_left = face_model(face_left_input)
    face_right = face_model(face_right_input)

    merged = merge([face_left, face_right], mode=lambda tensors: abs(tensors[0] - tensors[1]), output_shape=(512,))
    merged = Dropout(0.5)(merged)

    merged = Dense(256, init=init_dense)(merged)
    merged = BatchNormalization()(merged)
    merged = LeakyReLU(0.33)(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(1)(merged)
    merged = Activation("sigmoid")(merged)

    classification_model = Model(input=[face_left_input, face_right_input], output=merged)

    optimizer = Adam()

    print("Compiling model...")
    classification_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return classification_model, optimizer

def train_loop(identifier, model, optimizer, epoch_start, history, la_plotter,
               ia_train, ia_val, X_train, y_train, X_val, y_val):
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
        # interval=0 is required for small batch sizes
        progbar = generic_utils.Progbar(nb_examples_train, interval=0)
        #progbar = Progbar(nb_examples_train)

        for X_batch, Y_batch in flow_batches(X_train, y_train, ia_train,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True, train=True):
            bsize = X_batch[0].shape[0] # we dont use BATCH_SIZE here, because the
                                        # last batch might have less entries than BATCH_SIZE
            loss, acc = model.train_on_batch(X_batch, Y_batch)
            progbar.add(bsize, values=[("train loss", loss), ("train acc", acc)])
            loss_train_sum += (loss * bsize)
            acc_train_sum += (acc * bsize)

        # Validation loop
        # interval=0 is required for small batch sizes
        progbar = generic_utils.Progbar(nb_examples_val, interval=0)
        #progbar = Progbar(nb_examples_val)

        # Iterate over each batch in the validation data
        # and calculate loss and accuracy for each batch
        for X_batch, Y_batch in flow_batches(X_val, y_val, ia_val,
                                             batch_size=BATCH_SIZE_VAL,
                                             shuffle=False, train=False):
            bsize = X_batch[0].shape[0] # we dont use BATCH_SIZE here, because the
                                        # last batch might have less entries than BATCH_SIZE
            loss, acc = model.test_on_batch(X_batch, Y_batch)
            progbar.add(bsize, values=[("val loss", loss), ("val acc", acc)])
            loss_val_sum += (loss * bsize)
            acc_val_sum += (acc * bsize)

        # Calculate the loss and accuracy for this epoch
        # (averaged over all training data batches)
        loss_train = loss_train_sum / nb_examples_train
        acc_train = acc_train_sum / nb_examples_train
        loss_val = loss_val_sum / nb_examples_val
        acc_val = acc_val_sum / nb_examples_val

        history.add(epoch, loss_train=loss_train, loss_val=loss_val,
                    acc_train=acc_train, acc_val=acc_val)

        # Update plots with new data from this epoch
        # We start plotting _after_ the first epoch as the first one usually contains
        # a huge fall in loss (increase in accuracy) making it harder to see the
        # minor swings at epoch 1000 and later.
        if epoch > 0:
            la_plotter.add_values(epoch, loss_train=loss_train, loss_val=loss_val,
                                  acc_train=acc_train, acc_val=acc_val)

        # Save the history to a csv file
        if SAVE_CSV_FILEPATH is not None:
            csv_filepath = SAVE_CSV_FILEPATH.format(identifier=identifier)
            history.save_to_filepath(csv_filepath)

        # Save the weights and optimizer state to files
        swae = SAVE_WEIGHTS_AFTER_EPOCHS
        if swae and swae > 0 and (epoch+1) % swae == 0:
            print("Saving model...")
            save_model_weights(model, SAVE_WEIGHTS_DIR,
                               "{}.last.weights".format(identifier), overwrite=True)

def flow_batches(X_in, y_in, ia, batch_size=BATCH_SIZE, shuffle=False, train=False):
    """Uses the datasets (either train. or val.) and returns them batch by batch,
    transformed via the provided ImageAugmenter (ia).

    Args:
        X_in: Pairs of input images of shape (N, 2, 64, 64).
        y_in: Labels for the pairs of shape (N, 1).
        ia: ImageAugmenter to use.
        batch_size: Size of the batches to return.
        shuffle: Whether to shuffle (randomize) the order of the images before
            starting to return any batches.

    Returns:
        Batches, i.e. tuples of (X, y).
        Generator.
    """

    # Shuffle the datasets before starting to return batches
    if shuffle:
        # we copy X_in and y_in here, otherwise the original X_in and y_in
        # will be shuffled by numpy too.
        X = np.copy(X_in)
        y = np.copy(y_in)

        seed = random.randint(1, 10e6)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
    else:
        X = X_in
        y = y_in

    # Iterate over every possible batch and collect the examples
    # for that batch
    nb_examples = X.shape[0]
    batch_start = 0
    while batch_start < nb_examples:
        batch_end = batch_start + batch_size
        if batch_end > nb_examples:
            batch_end = nb_examples

        # extract all images of the batch from X
        batch = X[batch_start:batch_end]
        nb_examples_batch = batch.shape[0]

        # augment the images of the batch
        batch_img1 = batch[:, 0, ...] # left images
        batch_img2 = batch[:, 1, ...] # right images
        if train:
            batch_img1 = ia.augment_batch(batch_img1)
            batch_img2 = ia.augment_batch(batch_img2)
        else:
            batch_img1 = batch_img1 / 255.0
            batch_img2 = batch_img2 / 255.0

        # resize and merge the pairs of images to shape (B, 1, 32, 64), where
        # B is the size of this batch and 1 represents the only channel
        # of the image (grayscale).
        # X has usually shape (20000, 2, 64, 64, 3)
        height = X.shape[2]
        width = X.shape[3]
        nb_channels = X.shape[4] if len(X.shape) == 5 else 1
        X_batch_left = np.zeros((nb_examples_batch, height, width, nb_channels))
        X_batch_right = np.zeros((nb_examples_batch, height, width, nb_channels))
        #X_batch_left = np.zeros((nb_examples_batch, height, width))
        #X_batch_right = np.zeros((nb_examples_batch, height, width))
        for i in range(nb_examples_batch):
            # sometimes switch positions (left/right) of images during training
            if train and random.random() < 0.5:
                img1 = batch_img2[i]
                img2 = batch_img1[i]
            else:
                img1 = batch_img1[i]
                img2 = batch_img2[i]

            X_batch_left[i] = img1[:, :, np.newaxis]
            X_batch_right[i] = img2[:, :, np.newaxis]

        # Collect the y values of the batch
        y_batch = y[batch_start:batch_end]

        # from (B, H, W, C) to (B, C, H, W)
        X_batch_left = X_batch_left.transpose(0, 3, 1, 2)
        X_batch_right = X_batch_right.transpose(0, 3, 1, 2)

        yield [X_batch_left, X_batch_right], y_batch
        batch_start = batch_start + nb_examples_batch

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
    """
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 " \
                        "and _ are allowed.")
    if must_exist:
        if not identifier_exists(identifier):
            raise Exception("No model with identifier '{}' seems to " \
                            "exist.".format(identifier))

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

if __name__ == "__main__":
    main()
