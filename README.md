# About

This convolutional neural network estimates whether two images of human faces show the same or a different person. It is trained and tested on the [Labeled Faces in the Wild, greyscaled and cropped (LFWcrop_grey)](http://conradsanderson.id.au/lfwcrop/) dataset. Peak performance seems to be at about 90-91% accuracy.

# Requirements

* Python 2.7 (only tested in that version)
* scipy
* Numpy
* matplotlib
* [Keras](https://github.com/fchollet/keras)
* scikit-image (`sudo apt-get install python-skimage`)

# Usage

Install all requirements, download the [LFWcrop_grey](http://conradsanderson.id.au/lfwcrop/) dataset, extract it and clone the repository.
Then you can train the network using
```
python train.py name_of_experiment --images="/path/to/lfwcrop_grey/faces"
```
where
* `name_of_experiment` is a short identifier for your experiment (used during saving of files), e. g. "experiment_15_low_dropout".
* `/path/to/lfwcrop_grey/faces` is the path to the `/faces` subdirectory of your LFWcrop_grey dataset.

Training should take about an hour or less to reach high accuracy levels (assuming decent hardware).

You can test the network using
```
python test.py name_of_experiment --images="/path/to/lfwcrop_grey/faces"
```
which will output accuracy, error rate (1 - accuracy), recall, precision and f1 score for training, validation and test datasets. It will also plot/show pairs of images which resulted in false positives and false negatives (false positives: images of different people, but network thought they were the same).

# Architecture

The network uses two branches, one per image. Each branch applies a few convolutions and ends in a fully connected layer.
The outputs of both branches are then merged and further processed by another fully connected layer, before making the final yes-no-decision (whether both images show the same person).
All convolutions use leaky ReLUs (alpha=0.33) and no batch normalization.

![Architecture](images/architecture.png?raw=true "Architecture")

The used optimizer is Adam. All images are heavily augmented during training (rotation, translation, skew, ...).

# Results

![Example experiment training progress](images/example_experiment_lossacc.png?raw=true "Example experiment training progress")

The graph shows the training progress of the included example model over ~200 epochs. The red lines show the training set values (loss function and accuracy), while the blue lines show the validation set values. Light/Transparent lines are the real values, thicker/opaque lines are averages over the last 20 epochs.

## Examples of false positives (validation set)

False positives here are image pairs that show different people, but were classified by the network as showing the same person.

![False positives on the validation dataset](images/val_false_positives.png?raw=true "False positives on the validation dataset")


## Examples of false negatives (validation set)

False negatives here are image pairs that show the same people, but were classified by the network as showing different persons.

![False negatives on the validation dataset](images/val_false_negatives.png?raw=true "False negatives on the validation dataset")

# Dataset skew

The used dataset may seem quite big at first glance as it contains 13,233 images of 5,749 different people. However these images are highly unequally distributed over the different people. There are many people with barely any images and a few with lots of images:
* 4069 persons have 1 image.
* 779 persons have 2 images.
* 590 persons have 3-5 images.
* 168 persons have 6-10 images.
* 102 persons have 11-25 images.
* 35 persons have 26-75 images.
* 4 persons have 76-200 images.
* 2 persons have >=201 images.

In order to generate pairs of images showing the same person you need a person with at least two images (unless you allow pairs of images where both images are identical). So only 1,680 persons can possibly be used for such a pair. Furthermore, any image already added to a dataset (training, validation, test) *must not appear in another dataset*. If you don't stick to that rule, your results will be completely skewed as the net can just start to memorize the image->person mapping and you are essentially testing whether your network is a good memorizer, but not whether it has learned a generalizing rule to compare faces.

Because of these problems, the validation set is picked first, before the training set, so that it is less skewed (towards few people) than the training set. Additionally, a stratified sampling approach is used: First, a name is picked among all person names. Then, among all images of that person one image is picked. That is done two times for each pair. By doing this, early picked pairs should be less skewed (towards few people) than by just randomly picking images among all images (of all people). At the end, the validation set should be rather balanced, giving a better measurement of the true performance of the network. To decrease the resulting skew of the training set, the validation set size is kept small, at only 256 pairs, while the training set size is 40,000 pairs. All images that appear in the validation set are excluded from the picking process for the training set. (Sidenote: The picking process is also specifically designed in a way that exactly half of all picked pairs show the same person.)

The following plot shows roughly the count of images per person and dataset (only calculated on pairs of images showing the same person, as those show skew earlier than pairs with different persons):

![Dataset skew plot over training, val and test sets](images/example_experiment_skew.png?raw=true "Dataset skew plot over training, val and test sets")

Notice how the counts of images per person are very uniform in the validation set as it was picked first.

# License

[Creative Commons 4.0 Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
