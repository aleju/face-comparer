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
python train.py name_of_experiment --images="/path/to/lfwcrop_grey/faces" --dropout=0.5 --argmul=1.5
```
where 
* `name_of_experiment` is a short identifier for your experiment (used during saving of files), e. g. "experiment_15_low_dropout".
* `/path/to/lfwcrop_grey/faces` is the path to *the /faces subdirectory* of your LFWcrop_grey dataset.
* `dropout=0.5` sets the dropout rate of parts of the network to 0.5 (50%). (Not all layers have dropout.)
* `argmul=1.5` sets the image augmentation's strength multiplier (e.g. rotation and shifting of images) to a factor of 1.5, where 1.0 represents medium augmentation and 2.0 is rather strong augmentation (e. g. lots of rotation).

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
1. Input: Images of shape (1, 32, 64), where 1 is the only channel (greyscale), 32 is the image height and 64 is the width (2 images of width 32, concatenated next to each other).
2. Convolution 2D, 32 channels, 3x3 (full border mode, i.e. +2 pixels width and height)
3. Convolution 2D, 32 channels, 3x3 (valid border mode, i.e. -2 pixels width and height)
4. Max pooling 2D, 2x2
5. Convolution 2D, 64 channels, 3x3 (valid border mode)
6. Convolution 2D, 64 channels, 3x3 (valid border mode)
7. Dropout (0.5 by default)
8. Reshape from shape (64, 12, 28) to (64*4, 12*28 / 4), which roughly converts every pair of images into 4 slices (hairline, eyeline, noseline, mouthline).
9. BatchNormalization
10. GRU with 64 neurons per timestep and 64*4 timesteps (slices). Each timestep's results are fed to the next layer, creating (64*4) * 64 outputs.
11. Flatten / Reshape to shape (64*4*64,), i.e. one-dimensional vector
12. BatchNormalization
13. Dropout (0.5 by default)
14. Dense / Fully connected layer from 64*4*64 neurons to 1 neuron

Training was done with the *Adagrad* optimizer.

All activations are *Leaky ReLUs* with alpha 0.33, except for inner activations of the GRU (sigmoid and hard sigmoid, Keras default configuration) and the last dense layer (sigmoid).
An weak L2 norm of 0.000001 is applied to the last dense layer, just to prevent weights from going crazy.

The network performed best among many, many other tested architectures (though I'm hardware constrained, so I usually have to stop quite early). The GRU layer was chosen, because it seemed to perform just as good as dense layers and maxout layers, but required significantly less disk space to save. I also had the impression that it was less prone to overfitting and in theory should be better at counting how many slices of images seem to match (same person) than dense layers.

Other architectures and techniques tested (recalled from memory):
* Sigmoid, Tanh, ReLUs, PReLUs, LeakyReLUs(0.15), LeakyReLUs(0.66). They all all performed significantly worse (including ReLUs), except for LeakyReLUs(0.15), which only seemed to be slightly worse than LeakyReLUs(0.33). PReLUs were a good 5% worse. The order seemed to be roughly: LeakyReLUs(0.33) > LeakyReLUs(0.15) > PReLUs > ReLUs > tanh > sigmoid > LeakyReLUs(0.66).
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

# Results

![Example experiment training progress](images/example_experiment.png?raw=true "Example experiment training progress")

The graph shows the training progress of the included example model over ~1,900 epochs. The red lines show the training set values (loss function and accuracy), while the blue lines show the validation set values. Light/Transparent lines are the real values, thicker/opaque lines are averages over the last 20 epochs. The sudden decrease of the training set results at ~1,500 came from an increase of the augmentation strength (multiplier increased from 1.2 to 1.5).

The results of the included example model are:
* *Training set*: Accuracy 94.14%, Recall 0.9474, Precision 0.9362, F1 0.9417 (Support: 20,000 samples)
               | same   | different  | TRUTH
    ---------------------------------|------
         same  | 9474   | 646        |
    different  | 526    | 9354       |
    -----------|--------|------------|
    PREDICTION |
* *Validation set*: Accuracy 89.06%, Recall 0.9141, Precision 0.8731, F1 0.8931 (Support: 256)
               | same   | different  | TRUTH
    ---------------------------------|------
         same  | 117    | 17         |
    different  | 11     | 111        |
    -----------|--------|------------|
    PREDICTION |
* *Test set*: Accuracy 88.48%, Recall 0.9062, Precision 0.8689, F1 0.8872 (Support: 512)
               | same   | different  | TRUTH
    ---------------------------------|------
         same  | 232    | 35         |
    different  | 24     | 221        |
    -----------|--------|------------|
    PREDICTION |

The results of the validation set and test set are averaged over 50 augmented runs (so each image pair was augmented 50 times, 50 predictions were made, resulting in 50 probabilities between 0.0 and 1.0 and these values were averaged). Using 50 runs with augmentation instead of 1 run without resulted in very slightly improved results (might be down to luck).

Examples of false positives in the validation dataset (image pair contained different persons, but network thought they were the same person):

![False positives on the validation dataset](images/val_false_positives_cropped.png?raw=true "False positives on the validation dataset")


Examples of false negatives in the validation dataset (image pair contained the same person, but network thought the images showed different persons):

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

# Notes

* There is currently no function to compare images directly via their filepaths. Take a look at the test.py file to see roughly how that would be done.
* The current highscore on this task is reported by Facebook. They achieved around 93% accuracy. The main differences are: They used frontalization on the images, special convolutional layers with local parameters and trained first on classification (image to person name). They probably also had much more training images.