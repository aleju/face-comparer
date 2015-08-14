# -*- coding: utf-8 -*-
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
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adagrad
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU
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
#SAVE_WEIGHTS_AT_END = False
SHOW_PLOT_WINDOWS = True
Y_SAME = 1
Y_DIFFERENT = 0

np.random.seed(SEED)
random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier", help="A short name/identifier for your experiment, e.g. 'ex42b_more_dropout'.")
    parser.add_argument("--load", required=False, help="Identifier of a previous experiment that you want to continue (loads weights, optimizer state and history).")
    parser.add_argument("--dropout", required=False, help="Dropout rate (0.0 - 1.0) after the last conv-layer and after the GRU layer.")
    args = parser.parse_args()
    validate_identifier(args.identifier, must_exist=False)
    if args.load:
        validate_identifier(args.load)

    if identifier_exists(args.identifier):
        if args.identifier != args.load:
            agreed = ask_continue("[WARNING] Identifier '%s' already exists and is different from load-identifier '%s'. It will be overwritten. Continue? [y/n]" % (args.identifier, args.load))
            if not agreed:
                return

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

    print("Creating model...")
    model, optimizer = create_model(args.dropout)
    
    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    la_plotter = LossAccPlotter(save_to_filepath=SAVE_PLOT_FILEPATH.format(identifier=args.identifier))
    ia_train = ImageAugmenter(64, 64, hflip=True, vflip=False,
                              scale_to_percent=1.15, scale_axis_equally=False,
                              rotation_deg=25, shear_deg=8,
                              translation_x_px=7, translation_y_px=7)
    ia_train.pregenerate_matrices(10000)
    ia_val = ImageAugmenter(64, 64)

    if args.load:
        print("Loading previous model...")
        epoch_start, history = load_previous_model(args.load, model, optimizer, la_plotter)
    else:
        epoch_start = 0
        history = History()
    
    print("Training...")
    train_loop(args.identifier, model, optimizer, epoch_start, history, la_plotter, ia_train, ia_val, X_train, y_train, X_val, y_val)
    
    print("Finished.")

def validate_identifier(identifier, must_exist=True):
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 and _ are allowed.")
    if must_exist:
        if not identifier_exists(identifier):
            raise Exception("No model with identifier '{}' seems to exist.".format(identifier))

def identifier_exists(identifier):
    filepath = SAVE_CSV_FILEPATH.format(identifier=identifier)
    if os.path.isfile(filepath):
        return True
    else:
        return False

def ask_continue(message):
    choice = raw_input(message)
    while choice not in ["y", "n"]:
        choice = raw_input("Enter 'y' (yes) or 'n' (no) to continue.")
    return choice == "y"

def load_previous_model(identifier, model, optimizer, la_plotter):
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

class History(object):
    def __init__(self):
        #self.first_epoch = 1000 * 1000
        #self.last_epoch = -1
        self.epochs = []
        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []

    def add(self, epoch, loss_train=None, loss_val=None, acc_train=None, acc_val=None):
        self.epochs.append(epoch)
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)
        self.acc_train.append(acc_train)
        self.acc_val.append(acc_val)
        #self.first_epoch = min(self.first_epoch, epoch)
        #self.last_epoch = max(self.last_epoch, epoch)

    def add_all(self, start_epoch, loss_train, loss_val, acc_train, acc_val):
        last_epoch = start_epoch + len(loss_train)
        for epoch, lt, lv, at, av in zip(range(start_epoch, last_epoch+1), loss_train, loss_val, acc_train, acc_val):
            self.add(epoch, loss_train=lt, loss_val=lv, acc_train=at, acc_val=av)

    def save_to_filepath(self, csv_filepath):
        with open(csv_filepath, "w") as fp:
            csvw = csv.writer(fp, delimiter=",")
            # header row
            rows = [["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]]
            
            #data = data + [[r_e, r_tl, r_vl, r_ta, r_va] for r_e, r_tl, r_vl, r_ta, r_va in zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc)]
            rows.extend(zip(self.epochs, self.loss_train, self.loss_val, self.acc_train, self.acc_val))
            csvw.writerows(rows)

    def load_from_file(self, csv_filepath, last_epoch=None):
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
    """Uses the datasets for the branches of the convnet and the pca
    and returns them batch by batch, transformed via the data generators.
    
    This method is largely copied from kera's ImageDataGenerator.flow_batches().
    
    Args:
        X_conv: Images for the convnet branch (32x32)
        X_pca: Images for the pca branch (64x64)
        dg_conv: Datagenerator for the convnet branch (e.g. randomly
            rotates images and other stuff).
        dg_pca: Datagenerator for the pca branch.
        pca: The fitted scikit PCA object.
        batch_size: Size of the batches to return.
        shuffle: Whether to shuffle the images before starting to return
            any batches.
    Returns:
        Batches/Tuples of ([convnet batch, pca batch], y)
        (one by one via yield).
    """
    
    # Shuffle the datasets before starting to return batches
    if shuffle:
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
        
        # Collect the examples for the convnet-batch,
        # use the data generator to randomly transform each example
        batch_start_idx = b*batch_size
        batch = X[batch_start_idx:batch_start_idx + nb_samples]
        
        # augment images
        batch_img1 = batch[:, 0, ...] # left images
        batch_img2 = batch[:, 1, ...] # right images
        batch_img1 = ia.augment_batch(batch_img1)
        batch_img2 = ia.augment_batch(batch_img2)
        
        X_batch = np.zeros((nb_samples, 1, 32, 64))
        for i in range(nb_samples):
            # sometimes switch positions (left/right) of images during training
            if train and random.random() < 0.5:
                img1 = batch_img2[i]
                img2 = batch_img1[i]
            else:
                img1 = batch_img2[i]
                img2 = batch_img1[i]

            # downsize images
            # note: imresize projects the image into 0-255, even if it was 0-1.0 before
            img1 = misc.imresize(img1, (32, 32)) / 255.0
            img2 = misc.imresize(img2, (32, 32)) / 255.0
            X_batch[i] = np.concatenate((img1, img2), axis=1)
        
        # Collect the y values for the batch
        y_batch = np.copy(y[batch_start_idx:batch_start_idx + nb_samples])

        yield X_batch, y_batch

if __name__ == "__main__":
    main()
