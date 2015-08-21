# -*- coding: utf-8 -*-
"""Script to test the performance of trained models."""
from __future__ import absolute_import, division, print_function
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from train import validate_identifier, flow_batches, create_model
from utils.datasets import get_image_pairs, image_pairs_to_xy, plot_dataset_skew
from utils.saveload import load_weights
from libs.ImageAugmenter import ImageAugmenter

SEED = 42
TRAIN_COUNT_EXAMPLES = 20000
VALIDATION_COUNT_EXAMPLES = 256
TEST_COUNT_EXAMPLES = 512
SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/experiments"
SAVE_WEIGHTS_DIR = "%s/weights" % (SAVE_DIR)
SHOW_PLOT_WINDOWS = True

np.random.seed(SEED)
random.seed(SEED)

def main():
    """Main function, estimates the performance of a model on the training,
    validation und test datasets."""
    # handle arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier", help="Identifier of the experiment of " \
                                           "which to load the weights.")
    parser.add_argument("--images", required=True,
                        help="Filepath to the 'faces/' subdirectory in the " \
                             "'Labeled Faces in the Wild grayscaled and cropped' " \
                             "dataset.")
    parser.add_argument("--augmul", required=False,
                        help="Multiplicator for the augmentation " \
                             "(0.0=no augmentation, 1.0=normal aug., 2.0=rather " \
                             "strong aug.). Default is 1.5.")
    args = parser.parse_args()
    validate_identifier(args.identifier, must_exist=True)

    if not os.path.isdir(args.images):
        raise Exception("The provided filepath to the dataset seems to not exist.")

    # Load:
    #  1. Validation set,
    #  2. Training set,
    #  3. Test set
    # We will test on each one of them.
    # Results from training and validation set are already known, but will be shown
    # in more detail here.
    # Additionally, we need to load train and val datasets to make sure that no image
    # contained in them is contained in the test set.
    print("Loading validation set...")
    pairs_val = get_image_pairs(args.images, VALIDATION_COUNT_EXAMPLES,
                                pairs_of_same_imgs=False, ignore_order=True,
                                exclude_images=list(), seed=SEED, verbose=False)
    assert len(pairs_val) == VALIDATION_COUNT_EXAMPLES
    X_val, y_val = image_pairs_to_xy(pairs_val)

    print("Loading training set...")
    pairs_train = get_image_pairs(args.images, TRAIN_COUNT_EXAMPLES,
                                  pairs_of_same_imgs=False, ignore_order=True,
                                  exclude_images=pairs_val, seed=SEED, verbose=False)
    assert len(pairs_train) == TRAIN_COUNT_EXAMPLES
    X_train, y_train = image_pairs_to_xy(pairs_train)

    print("Loading test set...")
    pairs_test = get_image_pairs(args.images, TEST_COUNT_EXAMPLES,
                                 pairs_of_same_imgs=False, ignore_order=True,
                                 exclude_images=pairs_val+pairs_train, seed=SEED,
                                 verbose=True)
    assert len(pairs_test) == TEST_COUNT_EXAMPLES
    X_test, y_test = image_pairs_to_xy(pairs_test)
    print("")

    # Plot dataset skew
    print("Plotting dataset skew. (Only for pairs of images showing the same person.)")
    print("More unequal bars mean that the dataset is more skewed (towards very few people).")
    print("Close the chart to continue.")
    plot_dataset_skew(
        pairs_train, pairs_val, pairs_test,
        only_y_same=True,
        show_plot_windows=SHOW_PLOT_WINDOWS
    )

    print("Creating model...")
    model, _ = create_model(0.00)
    (success, last_epoch) = load_weights(model, SAVE_WEIGHTS_DIR, args.identifier)
    if not success:
        raise Exception("Could not successfully load model weights")
    print("Loaded model weights of epoch '%s'" % (str(last_epoch)))

    # If we just do one run over a set (training/val/test) we will not augment
    # the images (ia_noop). If we do multiple runs, we will augment images in
    # each run (ia).
    augmul = float(args.augmul) if args.augmul is not None else 1.50
    ia_noop = ImageAugmenter(64, 64)
    ia = ImageAugmenter(64, 64, hflip=True, vflip=False,
                        scale_to_percent=1.0 + (0.075*augmul),
                        scale_axis_equally=False,
                        rotation_deg=int(7*augmul),
                        shear_deg=int(3*augmul),
                        translation_x_px=int(3*augmul),
                        translation_y_px=int(3*augmul))

    # ---------------
    # Run the tests on the train/val/test sets.
    # We will do a standard testing with one run over each set.
    # We will also do a non-standard test for val and test set where we do
    # multiple runs over the same images and augment them each time. Then
    # we will average the predictions for each pair over the runs to come
    # to a final conclusion.
    # Using augmentation seems to improve the results very slightly (<1% difference).
    # ---------------

    # only 1 run for training set, as 10 or more runs would take quite long
    # when tested, 10 runs seemed to improve the accuracy by a tiny amount
    print("-------------")
    print("Training set results (averaged over 1 run)")
    print("-------------")
    evaluate_model(model, X_train, y_train, ia_noop, 1)
    print("")

    print("-------------")
    print("Validation set results (averaged over 1 run)")
    print("-------------")
    evaluate_model(model, X_val, y_val, ia_noop, 1)
    print("")

    print("-------------")
    print("Validation set results (averaged over 50 runs)")
    print("-------------")
    evaluate_model(model, X_val, y_val, ia, 50)

    print("-------------")
    print("Test set results (averaged over 1 run)")
    print("-------------")
    evaluate_model(model, X_test, y_test, ia_noop, 1)
    print("")

    print("-------------")
    print("Test set results (averaged over 50 runs)")
    print("-------------")
    evaluate_model(model, X_test, y_test, ia, 50)

    print("Finished.")

def evaluate_model(model, X, y, ia, nb_runs):
    """Evaluate a model on a chosen datasets over N runs.

    Tests deal with:
    - Accuracy (% correct)
    - Error (% wrong)
    - Recall
    - Precision
    - F1
    - Confusion Matrix

    Additionally, up to 20 false positive pairs will be plotted
    as well as up to 20 false negatives.

    If the number of runs N is equal to 1, then no augmentation will be applied.
    If the number of runs N is >= 1, then the images will be augmented and
      N predictions will be made for each pair of images. The predictions will
      be averaged and if the average is >0.5 then the prediction is 1,
      otherwise 0.

    Args:
        model: The model to use for the predictions
        X: Image pairs to predict on. (numpy array of shape (N, 2, 32, 32); float32.)
        y: Labels of the image pairs to predict on. (numpy array of shape N, 1; float32.)
        ia: ImageAugmenter to use.
    """
    # results contains counts of true/false predictions
    # [1][1] is true positive (truth: same, pred: same)
    # [1][0] is false negative (truth: same, pred: diff)
    # [0][1] is false positive (truth: diff, pred: same)
    # [0][0] is true negative (truth: diff, pred: diff)
    # where same = both images show the same person
    #       diff = the images show different people
    results = [[0, 0], [0, 0]]
    false_positives = [] # image pairs to plot later
    false_negatives = [] # same here

    # we augment if more than one run has been requested
    train_mode = False if nb_runs == 1 else True
    predictions = np.zeros((X.shape[0], nb_runs), dtype=np.float32)
    for run_idx in range(nb_runs):
        pair_idx = 0
        for X_batch, Y_batch in flow_batches(X, y, ia, shuffle=False, train=train_mode):
            Y_pred = model.predict_on_batch(X_batch)
            for i in range(Y_pred.shape[0]):
                predictions[pair_idx][run_idx] = Y_pred[i]
                pair_idx += 1

    predictions_prob = np.average(predictions, axis=1)

    for pair_idx in range(X.shape[0]):
        truth = int(y[pair_idx])
        prediction = 1 if predictions_prob[pair_idx] > 0.5 else 0
        results[truth][prediction] += 1

        img1 = X[pair_idx, 0, ...]
        img2 = X[pair_idx, 1, ...]
        img_pair = (img1, img2)

        if truth == 0 and prediction == 1:
            # 0 is channel 0 (grayscale images, only 1 channel)
            false_positives.append(img_pair)
        elif truth == 1 and prediction == 0:
            false_negatives.append(img_pair)

    tp = results[1][1]
    tn = results[0][0]
    fn = results[1][0]
    fp = results[0][1]

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Correct: %d (%.4f)" % (tp + tn, (tp + tn)/X.shape[0]))
    print("Wrong: %d (%.4f)" % (fp + fn, (fp + fn)/X.shape[0]))
    print("Recall: %.4f, Precision: %.4f, F1: %.4f, Support: %d" \
          % (recall, precision, f1, X.shape[0]))

    confusion_matrix = \
    """
              | same   | different  | TRUTH
    ---------------------------------
         same | {:<5}  | {:<5}      |
    different | {:<5}  | {:<5}      |
    ---------------------------------
    PREDICTION
    """
    print("Confusion Matrix (assuming Y=1 => same, Y=0 => different):")
    print(confusion_matrix.format(tp, fp, fn, tn))

    print("Showing up to 20 false positives (truth: diff, pred: same)...")
    show_image_pairs(false_positives[0:20])
    print("Showing up to 20 false negatives (truth: same, pred: diff)...")
    show_image_pairs(false_negatives[0:20])

def show_image_pairs(image_pairs):
    """Plot pairs of images.

    All pairs will be shown in the same window in two columns.

    Args:
        images_pairs: A list of (image1, image2) where the images are numpy
            arrays of shape (height, width) with pixel values.
    """

    # 2 columns clearly visible in the plot,
    # but internally handled as 6 columns
    # | image 1a | image 1b | gap | image 2a | image 2b | gap
    # | image 3a | image 3b | gap | image 4a | image 4b | gap
    # ...
    # placing a gap at the end of each line is not neccessary, but simplifies
    # the loop
    nb_cols = 6

    # we need at least one row of images
    # and additionally for every 6 cells filled (=>6 columns) another row
    nb_rows = 1 + int(len(image_pairs)*3 / nb_cols)

    fig = plt.figure(figsize=(6, 12))
    plot_number = 1 # index of the cell

    for image1, image2 in image_pairs:
        # place img1
        ax = fig.add_subplot(nb_rows, nb_cols, plot_number, xticklabels=[],
                             yticklabels=[])
        ax.set_axis_off()
        plt.imshow(image1, cmap=cm.Greys_r, aspect="equal")

        # place img2
        ax = fig.add_subplot(nb_rows, nb_cols, plot_number + 1, xticklabels=[],
                             yticklabels=[])
        ax.set_axis_off()
        plt.imshow(image2, cmap=cm.Greys_r, aspect="equal")

        plot_number += 3 # 2 images placed, 1 image gap

    plt.show()

if __name__ == "__main__":
    main()
