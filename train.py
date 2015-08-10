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
    model, optimizer = create_model()
    
    # Calling the compile method seems to mess with the seeds (theano problem?)
    # Therefore they are reset here (numpy seeds seem to be unaffected)
    # (Seems to still not make runs reproducible.)
    random.seed(SEED)

    la_plotter = LossAccPlotter(save_to_filepath=SAVE_PLOT_FILEPATH.format(identifier=args.identifier))

    if args.load:
        print("Loading previous model...")
        epoch_start, history = load_previous_model(args.load, model, optimizer, la_plotter)
    else:
        epoch_start = 0
        history = History()
    
    print("Training...")
    train_loop(args.identifier, model, optimizer, epoch_start, history, la_plotter)
    
    print("Finished.")

def validate_identifier(identifier):
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 and _ are allowed.")

def load_previous_model(continue_identifier, model, optimizer, la_plotter):
    history = History()
    
    # load optimizer state
    (success, last_epoch) = load_optimizer_state(optimizer, SAVE_OPTIMIZER_STATE_DIR, continue_identifier)
    
    # load weights
    (success, last_epoch) = load_weights(model, SAVE_WEIGHTS_DIR, continue_identifier)
    
    if not success:
        raise Exception("Cannot continue previous experiment, because no weights were saved (yet?).")
    
    # load history from csv file
    history = load_history(continue_identifier, SAVE_CSV_DIR)
    
    # update loss acc plotter
    la_plotter.values_loss_train = history.loss_train
    la_plotter.values_loss_val = history.loss_val
    la_plotter.values_acc_train = history.acc_train
    la_plotter.values_acc_val = history.acc_val
    
    return last_epoch, history

def load_history(identifier, save_history_dir):
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

def create_model():
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
    
    return model, optimizer

class History(object):
    def __init__(self):
        self.first_epoch = 1000 * 1000
        self.last_epoch = -1
        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []

    def add(self, epoch, loss_train=None, loss_val=None, acc_train=None, acc_val=None):
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)
        self.acc_train.append(acc_train)
        self.acc_val.append(acc_val)
        self.first_epoch = min(self.first_epoch, epoch)
        self.last_epoch = max(self.last_epoch, epoch)

    def add_all(self, start_epoch, loss_train, loss_val, acc_train, acc_val):
        last_epoch = start_epoch + len(loss_train)
        for epoch, lt, lv, at, av in zip(range(start_epoch, last_epoch+1), loss_train, loss_val, acc_train, acc_val):
            self.add(epoch, loss_train=lt, loss_val=lv, acc_train=at, acc_val=av)

    def save_to_file(self, csv_filepath):
        with open(csv_filepath, "w") as fp:
            csvw = csv.writer(fp, delimiter=",")
            # header row
            rows = [["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]]
            
            #data = data + [[r_e, r_tl, r_vl, r_ta, r_va] for r_e, r_tl, r_vl, r_ta, r_va in zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc)]
            rows.extend(zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc))
            csvw.writerows(rows)

def train_loop(identifier, model, optimizer, epoch_start, history, la_plotter):
    # Loop over each epoch, i.e. executes 20 times if epochs set to 20
    
    for epoch in range(epoch_start, EPOCHS):
        print("Epoch", epoch)
        
        loss_train_sum = 0
        loss_val_sum = 0
        acc_train_sum = 0
        acc_val_sum = 0
        
        # Training (validation later)
        progbar = generic_utils.Progbar(n_examples_train)
        
        # Iterate over each batch in the training data
        # and train the net on those examples
        for X_batch, Y_batch in flow_batches(X_train, y_train, pca, embedder, batch_size=cfg["batch_size"], shuffle=True, train=True):
            loss, acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(len(X_batch), values=[("train loss", loss), ("train acc", acc)])
            loss_train_sum += (loss * len(X_batch))
            acc_train_sum += (acc * len(X_batch))
        
        # Validation
        progbar = generic_utils.Progbar(n_examples_val)
        
        # Iterate over each batch in the validation data
        # and calculate loss and accuracy for each batch
        for X_batch, Y_batch in flow_batches(X_val, y_val, pca, embedder, batch_size=cfg["batch_size"], shuffle=False, train=False):
            loss, acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(len(X_batch), values=[("val loss", loss), ("val acc", acc)])
            loss_val_sum += loss
            acc_val_sum += acc

        # Calculate the loss and accuracy for this epoch
        # (averaged over all training data batches)
        loss_train = loss_train_sum / len(X_train)
        acc_train = acc_train_sum / len(X_train)
        loss_val = loss_val_sum / len(X_val)
        acc_val = acc_val_sum / len(X_val)
        
        history.add(epoch, loss_train=loss_train, loss_val=loss_val, acc_train=acc_train, acc_val=acc_val)
        
        # ---
        # Update plots with new data from this epoch
        # ---
        if epoch > 0:
            la_plotter.add_values(epoch, loss_train=loss_train, loss_val=loss_val, acc_train=acc_train, acc_val=acc_val)
        
        if cfg["save_csv_filepath"] is not None:
            with open(cfg["save_csv_filepath"], "w") as fp:
                csvw = csv.writer(fp, delimiter=",")
                rows = [["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]]
                #data = data + [[r_e, r_tl, r_vl, r_ta, r_va] for r_e, r_tl, r_vl, r_ta, r_va in zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc)]
                rows.extend(zip(range(epoch+1), stats_train_loss, stats_val_loss, stats_train_acc, stats_val_acc))
                csvw.writerows(rows)
        
        swae = cfg["save_weights_after_epochs"]
        if swae and swae > 0 and (epoch+1) % swae == 0:
            print("Saving model...")
            #save_model(model, cfg["save_model_dir"], model_name + ".at" + str(epoch), cfg["save_model_extension"], use_gzip=cfg["save_model_gzip"])
            #save_model_config(model, cfg["save_weights_dir"], model_name + ".at" + str(epoch) + ".config")
            save_model_weights(model, cfg["save_weights_dir"], model_name + ".at" + str(epoch) + ".weights")
            save_optimizer_state(optimizer, cfg["save_optimizer_state_dir"], model_name + ".at" + str(epoch) + ".optstate", overwrite=True)

if __name__ == "__main__":
    main()
