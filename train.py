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
    

if __name__ == "__main__":
    main()
