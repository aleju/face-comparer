# About

This convolutional neural network estimates whether two images of human faces show the same or a different person. It is trained and tested on the [Labeled Faces in the Wild, greyscaled and cropped (LFWcrop_grey)](http://conradsanderson.id.au/lfwcrop/) dataset. Peak performance seems to be at around 88-89% accuracy with some spikes reaching values above 90%.

# Requirements

* Scipy
* Numpy
* matplotlib
* [Keras](https://github.com/fchollet/keras)
* h5py
* Python 2.7 (only tested in that version)

# Usage

Install all requirements, download the LFWcrop_grey dataset, extract it and clone the repository.
Then you can train the network using
```
python train.py name_of_experiment --images="/path/to/lfwcrop_grey/faces" --dropout=0.5 --augmul=1.5
```
where 
* `name_of_experiment` is a short identifier for your experiment (used during saving of files), e. g. "experiment_15_low_dropout".
* `/path/to/lfwcrop_grey/faces` is the path to *the /faces subdirectory* of your LFWcrop_grey dataset.
* `dropout=0.5` sets the dropout rate of parts of the network to 0.5 (50%). (Not all layers have dropout.)
* `augmul=1.5` sets the image augmentation's strength multiplier (e.g. rotation and shifting of images) to a factor of 1.5, where 1.0 represents medium augmentation and 2.0 is rather strong augmentation (e.g. lots of rotation).

Training will take a few hours for good performance and around a day to reach peak performance (depending on hardware and RNG luck, around 1000 epochs are needed).

You can test the network using
```
python test.py name_of_experiment --images="/path/to/lfwcrop_grey/faces"
```
which will output accuracy, error rate (1 - accuracy), recall, precision and f1 score for training, validation and test datasets. It will also plot/show pairs of images which resulted in false positives and false negatives (false positives: images of different persons, but network thought they were the same).

# Pretrained weights

This repository comes with a pretrained network, which should achieve around 89% accuracy, depending on the dataset (obviously more on the training set). The identifier is `example_experiment`. You can test it via
```
python test.py example_experiment --images="/path/to/lfwcrop_grey/faces"
```

# Architecture

The neural network has the following architecture, layer by layer:

| Layer type | Configuration           | Notes |
| ---------- | ----------------------- | ----- |
| Input      |                         | Images of shape (1, 32, 64), where 1 is the only channel (greyscale), 32 is the image height and 64 is the width (2 images of width 32, concatenated next to each other). |
| Conv2D     | 32 channels, 3x3, full  | full border mode adds +2 pixels to image width and height |
| Conv2D     | 32 channels, 3x3, valid | valid border mode removes 2 pixels from image width and height |
| MaxPooling2D | 2x2                   | |
| Conv2D     | 64 channels, 3x3, valid | |
| Conv2D     | 64 channels, 3x3, valid | |
| Dropout    | 0.5                     | |
| Reshape    | to (64\*4, 12\*28 / 4)  | from shape (64, 12, 28); which roughly converts every pair of images into 4 slices (hairline, eyeline, noseline, mouthline) |
| BatchNormalization |                 | |
| GRU        | 64 internal neurons, return sequences | Output shape is (#slices, 64) |
| Flatten    | | Creates 1D-Vector of size (64\*4) \* 64 |
| BatchNormalization |                 | |
| Dropout    | 0.5                     | |
| Dense      | from (64\*4)\*64 to 1 neuron, L2=0.000001 | Weak L2 only to prevent weights from going crazy |

**Adagrad** was used as the optimizer.

All activations are **Leaky ReLUs** with alpha 0.33, except for inner activations of the GRU (sigmoid and hard sigmoid, Keras default configuration) and the last dense layer (sigmoid).


# Results

![Example experiment training progress](experiments/plots/example_experiment.png?raw=true "Example experiment training progress")

The graph shows the training progress of the included example model over ~1,900 epochs. The red lines show the training set values (loss function and accuracy), while the blue lines show the validation set values. Light/Transparent lines are the real values, thicker/opaque lines are averages over the last 20 epochs. The sudden decrease of the training set results at ~1,500 came from an increase of the augmentation strength (multiplier increased from 1.2 to 1.5).

**Training set** (20k pairs)

* Accuracy 94.14%
* Recall 0.9474
* Precision 0.9362
* F1 0.9417

           | same   | different  | <span style="font-weight:normal">TRUTH</span>
---------- | ------ | ---------- | -----
     same  | 9474   | 646        |
different  | 526    | 9354       |
---------- | ------ | ---------- |
PREDICTION |

**Validation set** (256 pairs)

* Accuracy 89.06%
* Recall 0.9141
* Precision 0.8731
* F1 0.8931

           | same   | different  | <span style="font-weight:normal">TRUTH</span>
---------- | ------ | ---------- | -----
     same  | 117    | 17         |
different  | 11     | 111        |
---------- | ------ | ---------- |
PREDICTION |

**Test set** (512 pairs)

* Accuracy 88.48%
* Recall 0.9062
* Precision 0.8689
* F1 0.8872

           | same   | different  | <span style="font-weight:normal">TRUTH</span>
---------- | ------ | ---------- | -----
     same  | 232    | 35         |
different  | 24     | 221        |
---------- | ------ | ---------- |
PREDICTION |

The results of the validation set and test set are averaged over 50 augmented runs (so each image pair was augmented 50 times, 50 predictions were made, resulting in 50 probabilities between 0.0 and 1.0 and these values were averaged). Using 50 runs with augmentation instead of 1 run without resulted in very slightly improved results (might be down to luck).

Examples of **false positives** in the *validation set* (image pair contained different persons, but network thought they were the same person):

![False positives on the validation dataset](images/val_false_positives_cropped.png?raw=true "False positives on the validation dataset")


Examples of **false negatives** in the *validation set* (image pair contained the same person, but network thought the images showed different persons):

![False negatives on the validation dataset](images/val_false_negatives_cropped.png?raw=true "False negatives on the validation dataset")

# Dataset skew

The used dataset may seem quite big at first glance as it contains 13,233 images of 5,749 different persons. However these images are highly unequally distributed over the different people. There are many people with barely any images and a few with lots of images:
* 4069 persons have 1 image.
* 779 persons have 2 images.
* 590 persons have 3-5 images.
* 168 persons have 6-10 images.
* 102 persons have 11-25 images.
* 35 persons have 26-75 images.
* 4 persons have 76-200 images.
* 2 persons have >=201 images.

In order to generate pairs of images showing the same person you need a person with at least two images (unless you allow pairs of images where both images are identical). So only 1,680 persons can possibly be used for such a pair. Furthermore, any image already added to a dataset (training, validation, test) *must not appear in another dataset*. If you don't stick to that rule, your results will be completely skewed as the net can just start to memorize the image->person mapping and you are essentially testing whether your network is a good memorizer, but not whether it has learned a generalizing rule to compare faces.

Because of these problems, the validation set is picked first, before the training set, so that it is less skewed (towards few people) than the training set. Additionally, a stratified sampling approach is used: First, a name is picked among all person names. Then, among all images of that person one image is picked. That is done two times for each pair. By doing this, early picked pairs should be less skewed (towards few people) than by just randomly picking images among all images (of all people). At the end, the validation set should be rather balanced, giving a better measurement of the true performance of the network. To decrease the resulting skew of the training set, the validation set size is kept small, at only 256 pairs, while the training set size is 20,000 pairs. All images that appear in the validation set are excluded from the picking process for the training set.

In `test.py` there is also a test set. It contains 512 pairs and is picked after training set (and hence also after the validation set). As a result, it is significantly more skewed than the validation set. The distribution of images per person at the start of the picking process for the test set is:
* 123 persons have 1 image.
* 11 persons have 2 images.
* 16 persons have 3-5 images.
* 5 persons have 6-10 images.
* 9 persons have 11-25 images.
* 3 persons have 26-75 images.
* 3 persons have 76-200 images.
* 1 persons have >=201 images.

The following plot shows roughly the count of images per person and dataset (only calculated on pairs of images showing the same person, as those have the biggest influence on the dataset skew):

![Dataset skew plot over training, val and test sets](images/skew_plot_example_experiment_cropped.png?raw=true "Dataset skew plot over training, val and test sets")

Notice how the counts of images per person are very uniform in the validation set as it was picked first.

# Comparison of activation functions

The following images show training runs *on 8,000 examples each* of an older model with different activation functions. LeakyReLU(0.33) performed best. (Red thin line: training set values, Red thick line: average over training set values (last 20 epochs), Blue thin line: validation set values, Blue thick line: average over validation set values (last 20 epochs)).

LeakyReLU(0.66):
![Model trained on 8k examples with LeakyReLUs at 0.66](images/m23r8k_lrelu066_cropped.jpg?raw=true "Model trained on 8k examples with LeakyReLUs at 0.66")

LeakyReLU(0.33):
![Model trained on 8k examples with LeakyReLUs at 0.33](images/m23r8k_lrelu_cropped.jpg?raw=true "Model trained on 8k examples with LeakyReLUs at 0.33")

LeakyReLU(0.15):
![Model trained on 8k examples with LeakyReLUs at 0.15](images/m23r8k_lrelu015_cropped.jpg?raw=true "Model trained on 8k examples with LeakyReLUs at 0.15")

PReLU:
![Model trained on 8k examples with PReLUs](images/m23r8k_prelu_cropped.jpg?raw=true "Model trained on 8k examples with PReLUs")

ReLU:
![Model trained on 8k examples with ReLU activation](images/m23r8k_relu_cropped.jpg?raw=true "Model trained on 8k examples with ReLU activation")

Tanh:
![Model trained on 8k examples with tanh activation](images/m23r8k_tanh_cropped.jpg?raw=true "Model trained on 8k examples with tanh activation")

Sigmoid:
![Model trained on 8k examples with sigmoid activation](images/m23r8k_sigmoid_cropped.jpg?raw=true "Model trained on 8k examples with sigmoid activation")


# Other architectures tested

The chosen network architecture performed best among many, many other tested architectures (though I'm hardware constrained, so I usually have to stop runs earlier than I would like to). The GRU layer was chosen, because it seemed to perform just as good as dense layers and maxout layers, but required significantly less disk space to save. I also had the impression that it was less prone to overfitting and in theory should be better at counting how many slices of images seem to match (same person) than dense layers.

Other architectures and techniques tested (recalled from memory):
* (As seen in above chapter:) Sigmoid, Tanh, ReLUs, PReLUs, LeakyReLUs(0.15), LeakyReLUs(0.66). They all all performed significantly worse (including ReLUs), except for LeakyReLUs(0.15), which only seemed to be slightly worse than LeakyReLUs(0.33). PReLUs were a good 5% worse. The order seemed to be roughly: LeakyReLUs(0.33) > LeakyReLUs(0.15) > PReLUs > ReLUs > tanh > sigmoid > LeakyReLUs(0.66).
* Larger images of size 64x64 instead of 32x32. Seemed to only increase overfitting and training time per epoch.
* Using not the original images, but instead images after a canny edge detector was applied (i.e. black and white images, where edges are white). Performed significantly worse.
* Feeding the images in shape (2, 32, 32) into the network (one image with 2 channels, where each channel is one of the two images) instead of (1, 32, 64). Performed significantly worse (5-10% val. accuracy lost).
* Feeding the images in shape (1, 64, 32) where the first row contained the left half of image1 and the right half of image2, analogous for the second row. Performed worse.
* Using Adam optimizer. Seemed to increase overfitting.
* Various other convolution layer architectures. Examples would be: 4 channels into 8, 16, ...; growing the channel number up to 128 channels; using larger kernels (5x5). All of these architectures seemed to perform slightly or significantly worse. The ones ending in 128 channels routinely overfitted.
* Using only full border mode on the convolutional layers. One run seemed to end up a bit better (~1%), but a later one didn't reproduce that, so i assume that it doesn't really help.
* LSTM instead of GRU. Seemed to perform worse, but not thoroughly tested.
* No BatchNormalization. The network failed to learn anything. The last normalization layer after the GRU seems to be simply neccessary.
* Dropout between Conv-Layers. Seemed to only worsen the results.
* Lower Dropout rates than 0.5. Performs worse (around 1-3% val. accuracy).
* Higher Dropout rates than 0.5. Seemed to help.
* Gaussian Noise and Gaussian Dropout between the layers. Seemed to only worsen the results.
* Reshaping to 64 images instead of 64*4 slices. Seemed to worsen results.
* Dense layers and maxout layers instead of GRU. Seemed to not improve anything at higher hdd-space requirement.


# Notes

* There is currently no function to compare images directly via their filepaths. Take a look at the test.py file to see roughly how that would be done.
* The current highscore on this task is reported by Facebook. They achieved around 93% accuracy. The main differences are: They used frontalization on the images, special convolutional layers with local parameters and trained first on classification (image to person name). They probably also had much more training images.

# License

[Creative Commons 4.0 Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/)