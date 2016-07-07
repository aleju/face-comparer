# -*- coding: utf-8 -*-
"""The main training file.
Traings a neural net to compare pairs of faces with each other.

Call this file in the following way:

    python train.py name_of_experiment --images="/path/to/lfwcrop_grey/faces"

or more complex:

    python train.py name_of_experiment
           --images="/path/to/lfwcrop_grey/faces"
           --load="old_experiment_name"
           --dropout=0.5
           --augmul=1.5

where
    name_of_experiment:
        Is the name of this experiment, used when saving data, e.g. "exp5_more_dropout".
    --load="old_experiment_name":
        Is the name of an old experiment to continue. Must have the identical
        network architecture and optimizer as the new network.
    --dropout=0.5:
        Dropout strength to use for the last two dropout layers.
    --augmul=1.5:
        Augmentation strength to use when augmenting images (e.g. rotation, shift).
        0.5 is weak, 1.0 is normal, 1.5+ is strong.
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
from keras.optimizers import Adagrad, Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU
from keras.utils import generic_utils

from libs.ImageAugmenter import ImageAugmenter
from libs.laplotter import LossAccPlotter
from utils.saveload import load_previous_model, save_model_weights, save_optimizer_state
from utils.datasets import get_image_pairs, image_pairs_to_xy, plot_dataset_skew
from utils.History import History

SEED = 42
TRAIN_COUNT_EXAMPLES = 20000
VALIDATION_COUNT_EXAMPLES = 256
EPOCHS = 1000 * 1000
BATCH_SIZE = 64
SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/experiments"
SAVE_PLOT_FILEPATH = "%s/plots/{identifier}.png" % (SAVE_DIR)
SAVE_DISTRIBUTION_PLOT_FILEPATH = "%s/plots/{identifier}_dataset_skew.png" % (SAVE_DIR)
SAVE_CSV_FILEPATH = "%s/csv/{identifier}.csv" % (SAVE_DIR)
SAVE_WEIGHTS_DIR = "%s/weights" % (SAVE_DIR)
SAVE_OPTIMIZER_STATE_DIR = "%s/optstate" % (SAVE_DIR)
SAVE_WEIGHTS_AFTER_EPOCHS = 1
SHOW_PLOT_WINDOWS = True

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
    parser.add_argument("--dropout", required=False,
                        help="Dropout rate (0.0 - 1.0) after the last " \
                             "conv-layer and after the GRU layer. Default " \
                             "is 0.5.")
    parser.add_argument("--augmul", required=False,
                        help="Multiplicator for the augmentation " \
                             "(0.0=no augmentation, 1.0=normal aug., " \
                             "2.0=rather strong aug.). Default is 1.5.")
    args = parser.parse_args()
    validate_identifier(args.identifier, must_exist=False)

    if not os.path.isdir(args.images):
        raise Exception("The provided filepath to the dataset seems to not exist.")

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

    if args.augmul is None:
        args.augmul = 1.5

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
    X_val, y_val = image_pairs_to_xy(pairs_val)
    X_train, y_train = image_pairs_to_xy(pairs_train)

    # Plot dataset skew
    print("Plotting dataset skew. (Only for pairs of images showing the same " \
          "person.)")
    print("More unequal bars mean that the dataset is more skewed (towards very " \
          "few people).")
    print("Close the chart to continue.")
    plot_dataset_skew(
        pairs_train, pairs_val, [],
        only_y_same=True,
        show_plot_windows=SHOW_PLOT_WINDOWS,
        save_to_filepath=SAVE_DISTRIBUTION_PLOT_FILEPATH.format(identifier=args.identifier)
    )

    # initialize the network
    print("Creating model...")
    model, optimizer = create_model(args.dropout)

    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    # -------------------
    # Training loop part
    # -------------------
    # initialize the plotter for loss and accuracy
    sp_fpath = SAVE_PLOT_FILEPATH.format(identifier=args.identifier)
    la_plotter = LossAccPlotter(save_to_filepath=sp_fpath)

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
        epoch_start, history = \
            load_previous_model(args.load, model, optimizer, la_plotter,
                                SAVE_OPTIMIZER_STATE_DIR, SAVE_WEIGHTS_DIR,
                                SAVE_CSV_FILEPATH)
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
    """Creates, compiles and returns the neural net with an Adagrad optimizer object.

    Structure of the network:
        1. Input: (BatchSize, 1, 32, 64) -- two grayscale images next to each other
        2. Conv2D, 32 output planes, kW/kH=3/3, Activation=LeakyReLU(0.33)
        3. Conv2D, 32 output planes, kW/kH=3/3, Activation=LeakyReLU(0.33)
        4. Max pooling kW/kH=2/2
        5. Conv2D, 64 output planes, kW/kH=3/3, Activation=LeakyReLU(0.33)
        6. Conv2D, 64 output planes, kW/kH=3/3, Activation=LeakyReLU(0.33)
        7. Dropout
        8. Reshape conv layer results from 64 images to 64 times 4 slices of images
        9. Normalize
        10. GRU (processes each slice, 64 nodes per timestep)
        11. Flatten GRU results to 1D vector
        12. Normalize
        13. Dropout
        14. Fully connected layer from (64*4)*64 to 1 neuron, Activation=sigmoid
        15. Output: 1 value between 0 and 1

    Args:
        dropout: Dropout probability to use after the conv-layers and after
            the GRU.
    Returns:
        Tuple of (neural net, optimizer), where the optimizer is Adagrad.
    """
    dropout = float(dropout) if dropout is not None else 0.50
    print("Dropout will be set to {}".format(dropout))

    model = Sequential()

    # Note: using dropout(0.00) in the network enables us to load an old
    # network's weights and set the dropout at these positions to >0.00 by
    # changing the code here. If we would not have these layers, adding them
    # (with p>0.00) and then loading an old network would result in a layer
    # number mismatch and the weights could no longer be associated properly.

    # -----
    # Convolutional Layers
    # -----
    # 32 x 32+2 x 64+2 = 32x34x66
    model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(1, 32, 64)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))

    # 32 x 34-2 x 66-2 = 32x32x64
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))

    # 32 x 32/2 x 64/2 = 32x16x32
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 64 x 16-2 x 32-2 = 64x14x30
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.00))

    # 64 x 14-2 x 30-2 = 64x12x28
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Dropout(dropout))

    # -----
    # Reshape the output of the conv layers to 64 times 4 slices (roughly hairline,
    # eyeline, noseline, mouthline)
    # -----
    # 64x12x28 = 64x336 = 21504
    # In 64*4 slices: 64*4 x 336/4 = 256x84
    model.add(Reshape((64*4, int((16*32)/4))))
    #model.add(BatchNormalization((64*4, int(336/4))))

    # -----
    # GRU / Recurrent Layer
    # processes each slice on its own
    # -----
    # GRU with 64*4 timesteps, each returning 64 values
    #model.add(GRU(336/4, 64, return_sequences=True))
    model.add(GRU(64, return_sequences=True))
    model.add(Flatten())
    #model.add(BatchNormalization((64*(64*4),)))
    model.add(Dropout(dropout))

    # -----
    # Output layer
    # We only need one output neuron, therefore a softmax is unneccessary
    # -----
    #model.add(Dense(64*(64*4), 1, init="glorot_uniform", W_regularizer=l2(0.000001)))
    model.add(Dense(1, W_regularizer=l2(0.000001)))
    model.add(Activation("sigmoid"))

    #optimizer = Adagrad()
    optimizer = Adam()

    print("Compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model, optimizer

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
        progbar = generic_utils.Progbar(nb_examples_train)

        for X_batch, Y_batch in flow_batches(X_train, y_train, ia_train,
                                             shuffle=True, train=True):
            loss, acc = model.train_on_batch(X_batch, Y_batch)
            progbar.add(len(X_batch), values=[("train loss", loss), ("train acc", acc)])
            loss_train_sum += (loss * len(X_batch))
            acc_train_sum += (acc * len(X_batch))

        # Validation loop
        progbar = generic_utils.Progbar(nb_examples_val)

        # Iterate over each batch in the validation data
        # and calculate loss and accuracy for each batch
        for X_batch, Y_batch in flow_batches(X_val, y_val, ia_val,
                                             shuffle=False, train=False):
            loss, acc = model.test_on_batch(X_batch, Y_batch)
            progbar.add(len(X_batch), values=[("val loss", loss), ("val acc", acc)])
            loss_val_sum += (loss * len(X_batch))
            acc_val_sum += (acc * len(X_batch))

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
            save_optimizer_state(optimizer, SAVE_OPTIMIZER_STATE_DIR,
                                 "{}.last.optstate".format(identifier),
                                 overwrite=True)

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
        # of the image (grayscale).
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
