# -*- coding: utf-8 -*-
"""File with helper functions to load test/validation/training sets from
the root dataset of Labeled Faces in the Wild, grayscaled and cropped (which
must already be on the hard drive).

The main function to use is get_image_pairs().
"""
from __future__ import absolute_import, division, print_function
import random
import re
import os
from collections import defaultdict
from scipy import misc
import numpy as np

Y_SAME = 1
Y_DIFFERENT = 0
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

class ImageFile(object):
    """Object to model one image file of the dataset.
    Example:
      image_file = ImageFile("~/lfw-gc/faces/", "Arnold_Schwarzenegger_001.pgm")
    """
    def __init__(self, directory, name):
        """Initialize the ImageFile object.
        Args:
            directory: Directory of the file.
            name: Full filename, e.g. 'foo.txt'."""
        self.filepath = os.path.join(directory, name)
        self.filename = name
        self.person = filepath_to_person_name(self.filepath)
        self.number = filepath_to_number(self.filepath)
    
    def get_content(self):
        """Returns the content of the image (pixel values) as a numpy array.
        
        Returns:
            Content of image as numpy array (dtype: uint8).
            Should have shape (height, width) as the images are grayscaled."""
        return misc.imread(self.filepath)

class ImagePair(object):
    """Object that models a pair of images, used during training of the neural net.
    Use the instance variables
        - 'same_person' to determine whether both images of the pair
           show the same person and
        - 'same_image' whether both images of the pair are identical.
    """
    def __init__(self, image1, image2):
        """Create a new ImagePair object.
        Args:
            image1: ImageFile object of the first image in the pair.
            image2: ImageFile object of the second image in the pair.
        """
        self.image1 = image1
        self.image2 = image2
        self.same_person = (image1.person == image2.person)
        self.same_image = (image1.filepath == image2.filepath)

    def get_key(self, ignore_order):
        """Return a key to represent this pair, e.g. in sets.

        Returns:
            A (string-)key representing this pair.
        """
        # if ignore_order then (image1,image2) == (image2,image1)
        # therefore, the key used to check if a pair already exists must then
        # catch both cases (A,B) and (B,A), i.e. it must be sorted to always
        # be (A,B)
        # Could probably use tuples here as keys too.
        fps = [self.image1.filepath, self.image2.filepath]
        if ignore_order:
            key = "$$$".join(sorted(fps))
        else:
            key = "$$$".join(fps)
        return key

    def get_contents(self):
        """Returns the contents (pixel values) of both images of the pair as one numpy array.
        Returns:
            Numpy array of shape (2, height, width) with dtype uint8.
        """
        return np.array([self.image1.get_content(), self.image2.get_content()], dtype=np.uint8)

def filepath_to_person_name(fp):
    """Extracts the name of a person from a filepath.
    Obviously only works with the file naming used in the LFW-GC dataset.

    Args:
        fp: The full filepath of the file.
    Returns:
        Name of the person.
    """
    last_slash = fp.rfind("/")
    if last_slash is None:
        return fp[0:fp.rfind("_")]
    else:
        return fp[last_slash+1:fp.rfind("_")]

def filepath_to_number(filepath):
    """Extracts the number of the image from a filepath.

    Each person in the dataset may have 1...N images associated with him/her,
    which are then numbered from 1 to N in the filepath. This function returns
    that number.

    Args:
        filepath: The full filepath of the file.
    Returns:
        Number of that image (among all images of that person).
    """
    return int(re.sub(r"[^0-9]", "", filepath))

def get_image_files(dataset_filepath, exclude_images=None):
    """Loads all images sorted by filenames and returns them as ImageFile Objects.
    
    Args:
        dataset_filepath: Path to the 'faces/' subdirectory of the dataset (Labeled
            Faces in the Wild, grayscaled and cropped).
        exclude_images: List of ImageFile objects to exclude from the list to
            return, e.g. because they are already used for another set of
            images (training, validation, test).
    Returns:
        List of ImageFile objects containing all images in the dataset filepath,
        except for the ones in exclude_images.
    """
    if not os.path.isdir(dataset_filepath):
        raise Exception("Images filepath '%s' of the dataset seems to not exist or is not a directory." % (dataset_filepath,))

    images = []
    exclude_images = exclude_images if exclude_images is not None else set()
    exclude_filenames = set()
    for image_file in exclude_images:
        exclude_filenames.add(image_file.filename)

    for directory, subdirs, files in os.walk(dataset_filepath):
        for name in files:
            if name.endswith(".pgm"):
                if name not in exclude_filenames:
                    images.append(ImageFile(directory, name))
    images = sorted(images, key=lambda image: image.filename)
    return images

def get_image_pairs(dataset_filepath, nb_max, pairs_of_same_imgs=False, ignore_order=True, exclude_images=list(), seed=None, verbose=False):
    """Creates a list of ImagePair objects from images in the dataset directory.
    
    This is the main method intended to load training/validation/test datasets.
    
    The images are all expected to come from the
        Labeled Faces in the Wild, grayscaled and cropped (sic!)
    dataset.
    Their names must be similar to:
        Adam_Scott_0002.pgm
        Kalpana_Chawla_0002.pgm
    
    Note: This function may currently run endlessly if nb_max is set too higher
    (above maximum number of possible pairs of same or different persons, whichever
    is lower - that number however is pretty large).
    
    Args:
        images_filepath: Path to the 'faces/' subdirectory of the dataset (Labeled
            Faces in the Wild, grayscaled and cropped).
        nb_max: Maximum number of image pairs to return. If there arent enough
            possible pairs, less pairs will be returned.
        pairs_of_same_imgs: Whether pairs of images may be returned, where
            filepath1 == filepath2. Notice that this may return many
            pairs of same images as many people only have low amounts of
            images. (Default is False.)
        ignore_order: Defines whether (image1, image2) shall be
            considered identical to (image2, image1). So if one
            of them is already added, the other pair wont be added any more.
            Setting this to True will result in less possible but more
            diverse pairs of images. (Default is True.)
        exclude_images: List of ImagePair objects with images that will be
            excluded from the result, i.e. no image that is contained in any pair
            in that list will be contained in any pair of the result of this
            function. Useful to fully separate validation and training sets.
        seed: A seed to use at the start of the function.
        verbose: Whether to print messages with statistics about the dataset
            and the collected images.
    Returns:
        List of ImagePair objects.
    """
    if seed is not None:
        state = random.getstate() # used right before the return
        random.seed(seed)
    
    # validate dataset directory
    if not os.path.isdir(dataset_filepath):
        raise Exception("Images filepath '%s' of the dataset seems to not exist or is not a directory." % (dataset_filepath,))
    
    # Build set of images to not use in image pairs (because they have
    # been used previously)
    exclude_images = set([img_pair.image1 for img_pair in exclude_images]
                         + [img_pair.image2 for img_pair in exclude_images])

    # load metadata of all images as ImageFile objects (except for the excluded ones)
    images = get_image_files(dataset_filepath, exclude_images=exclude_images)
    
    # build a mapping person=>images[]
    # this will make it easier to do stratified sampling of images
    images_by_person = defaultdict(list)
    for image in images:
        images_by_person[image.person].append(image)
    
    nb_img = len(images)
    nb_people = len(images_by_person)
    
    # Show some statistics about the dataset
    if verbose:
        print("Found %d images in filepath, resulting in theoretically max k*(k-1)=%d ordered or (k over 2)=k(k-1)/2=%d unordered pairs." % (nb_img, nb_img*(nb_img-1), nb_img*(nb_img-1)/2))
        print("Found %d different persons" % (nb_people,))
        print("In total...")
        print(" {:>7} persons have 1 image.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) == 1])))
        print(" {:>7} persons have 2 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) == 2])))
        print(" {:>7} persons have 3-5 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) >= 3 and len(fps) <= 5])))
        print(" {:>7} persons have 6-10 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) > 5 and len(fps) <= 10])))
        print(" {:>7} persons have 11-25 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) > 10 and len(fps) <= 25])))
        print(" {:>7} persons have 26-75 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) > 25 and len(fps) <= 75])))
        print(" {:>7} persons have 76-200 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) > 75 and len(fps) <= 200])))
        print(" {:>7} persons have >=201 images.".format(len([name for name, fps in images_by_person.iteritems() if len(fps) > 200])))
    
    # Create lists
    #  a) of all names of people appearing in the dataset
    #  b) of all names of people appearing in the dataset
    #     with at least 2 images
    names = []
    names_gte2 = []
    for person_name, images in images_by_person.iteritems():
        names.append(person_name)
        if len(images) >= 2:
            names_gte2.append(person_name)
    
    # Calculate maximum amount of possible pairs of images showing the
    # same person (not identical with "good" pairs, e.g. may be 10,000
    # times Arnold Schwarzenegger)
    if verbose:
        sum_avail_ordered = 0
        sum_avail_unordered = 0
        for name in names_gte2:
            k = len(images_by_person[name])
            sum_avail_ordered += k*(k-1)
            sum_avail_unordered += k*(k-1)/2
        print("Can collect max %d ordered and %d unordered pairs of images that show the _same_ person." % (sum_avail_ordered, sum_avail_unordered))
    
    # ---
    # Build pairs of images
    # 
    # We use stratified sampling over the person to sample images.
    # So we pick first a name among all available person names and then
    # randomly select an image of that person. (In contrast to picking a random
    # image among all images of all persons.) This makes the distribution of
    # the images more uniform over the persons. (In contrast to having a very
    # skewed distribution favoring much more people with many images.)
    # ---
    
    # result
    pairs = []
    
    # counters
    # only nb_added is really needed, we other ones are for print-output
    # in verbose mode
    nb_added = 0
    nb_same_p_same_img = 0 # pairs of images of same person, same image
    nb_same_p_diff_img = 0 # pairs of images of same person, different images
    nb_diff = 0
    
    # set that saves identifiers for pairs of images that have
    # already been added to the result.
    added = set() 
    
    # -------------------------
    # y = 1 (pairs with images of the same person)
    # -------------------------
    while nb_added < nb_max // 2:
        # pick randomly two images and make an ImagePair out of them
        person = random.choice(names_gte2)
        image1 = random.choice(images_by_person[person])
        if pairs_of_same_imgs:
            image2 = random.choice(images_by_person[person])
        else:
            image2 = random.choice([image for image in images_by_person[person] if image != image1])
        
        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)
        
        # add the ImagePair to the output, if the same pair hasn't been already
        # picked
        if key not in added:
            pairs.append(pair)
            nb_added += 1
            nb_same_p_same_img += 1 if pair.same_image else 0
            nb_same_p_diff_img += 1 if not pair.same_image else 0
            # log this pair as already added (dont add it a second time)
            added.add(key)
    
    # -------------------------
    # y = 0 (pairs with images of different persons)
    # -------------------------
    while nb_added < nb_max:
        # pick randomly two different persons names to sample each one image from
        person1 = random.choice(names)
        person2 = random.choice([person for person in names if person != person1])
        
        # we dont have to check here whether the images are the same,
        # because they come from different persons
        image1 = random.choice(images_by_person[person1])
        image2 = random.choice(images_by_person[person2])
        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)
        
        # add the ImagePair to the output, if the same pair hasn't been already
        # picked
        if key not in added:
            pairs.append(pair)
            nb_added += 1
            nb_diff += 1
            # log this pair as already added (dont add it a second time)
            added.add(key)
    
    # Shuffle the created list
    random.shuffle(pairs)
    
    # Print some statistics
    if verbose:
        print("Collected %d pairs of images total." % (nb_added,))
        print("Collected %d pairs of images showing the same person (%d are pairs of identical images)." % (nb_same_p_same_img + nb_same_p_diff_img, nb_same_p_same_img))
        print("Collected %d pairs of images showing different persons." % (nb_diff,))
    
    # reset the RNG to the state that it had before calling the method
    if seed is not None:
        random.setstate(state) # state was set at the start of this function
    
    return pairs

def image_pairs_to_xy(image_pairs):
    """Converts a list of ImagePair objects to X (array of pixel values) and
    Y (labels) to use during training/testing.
    
    Args:
        image_pairs: List of ImagePair objects.
    Returns:
        Tuple of X and Y, where X is a numpy array of dtype uint8 with
        shape (N, 2, height, width) containing pixel values of N pairs and
        Y is a numpy array of dtype float32 with shape (N, 1) containg
        the 'same person'/'different person' information.
    """
    X = np.zeros((len(image_pairs), 2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    y = np.zeros((len(image_pairs),), dtype=np.float32)
    
    for i, pair in enumerate(image_pairs):
        X[i] = pair.get_contents()
        y[i] = Y_SAME if pair.same_person else Y_DIFFERENT
    
    return X, y

def plot_dataset_skew(pairs_test, pairs_val, pairs_train, only_same=True, show_plot_windows=True, save_to_filepath=None):
    #nrows = sum([1 for ds in [pairs_test, pairs_val, pairs_train] if len(ds) > 0])
    nrows = 3
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=nrows, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.5)
    color = "b"
    bars_width = 0.2
    
    def plot_one_chart(ax, pairs, dataset_name, n_highest=250, n_highest_legend=15, first=False):
        name_to_images = defaultdict(list)
        for pair in pairs:
            if not only_same or (only_same and pair.same_person):
                name_to_images[pair.image1.person].append(pair.image1)
                name_to_images[pair.image2.person].append(pair.image2)
        
        names_with_counts = [(name, len(images)) for name, images in name_to_images.iteritems()]
        names_with_counts.sort(key=lambda name_with_count: name_with_count[1], reverse=True)
        names_with_counts = names_with_counts[0:n_highest]
        only_counts = np.array([count for name, count in names_with_counts])
        
        bars_positions = np.arange(len(names_with_counts))
        bars_names = [name for name, count in names_with_counts]
        bars_names_short = [re.sub(r"[^A-Z]", "", name) for name, count in names_with_counts]
        bars_values = only_counts
        
        bars_test = ax.bar(bars_positions, bars_values, bars_width, color=color)
        ax.set_ylabel("Count of images")
        ax.set_xlabel("Person name")
        if first:
            ax.set_title("Person-Count relation in dataset '%s'\n"
                        "A higher bar means that more images of that "
                        "person appear in this dataset\n"
                        "(only y value=%s, only highest %d persons)" % (dataset_name, str(only_y_value), n_highest))
        else:
            ax.set_title("Dataset '%s'" % (dataset_name,))
        
        ax.set_xticks(bars_positions + bars_width)
        ax.set_xticklabels(tuple(bars_names_short), rotation=90, size="x-small")
        
        name_translation = zip(bars_names_short, bars_names)
        text_arr1 = [short + "=" + full for (short, full) in name_translation][0:n_highest_legend]
        text_arr2 = []
        linebreak_every_n = 7
        for i, item in enumerate(text_arr1):
            # add linebreak after 10 names, but not if the name is the
            # last one shown
            if (i+1) % linebreak_every_n == 0 and (i+1) < len(text_arr1):
                text_arr2.append(item + "\n")
            else:
                text_arr2.append(item)
        textstr = " ".join(text_arr2)
        textstr += " (+%d others shown of total %d persons)" % (len(bars_names) - n_highest_legend, len(name_to_images))
        
        mean = np.mean(only_counts)
        if len(images) > 0:
            textstr += " (median=%.1f, mean=%.1f, std=%.2f)" % (np.median(only_counts), np.mean(only_counts), np.std(only_counts))
        else:
            textstr += " (median=%.1f, mean=%.1f, std=%.2f)" % (0, 0, 0)
        
        ax.text(0.3, 0.96, textstr, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=dict(alpha=0.5))
    
    plot_one_chart(ax1, fps_test, "test", first=True)
    plot_one_chart(ax2, fps_val, "validation")
    plot_one_chart(ax3, fps_train, "train")
    
    if save_to_filepath:
        fig.savefig(save_to_filepath)
        
    if show_plot_windows:
        plt.show()
