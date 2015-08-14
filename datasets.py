# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import random
from scipy import misc
import os
from collections import defaultdict
import numpy as np
import re

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def filepath_to_person_name(fp):
    last_slash = fp.rfind("/")
    if last_slash is None:
        return fp[0:fp.rfind("_")]
    else:
        return fp[last_slash+1:fp.rfind("_")]

def filepath_to_number(filepath):
    return int(re.sub(r"[^0-9]", "", filepath))

class ImageFile(object):
    def __init__(self, directory, name):
        self.filepath = os.path.join(directory, name)
        self.filename = name
        self.person = filepath_to_person_name(self.filepath)
        self.number = filepath_to_number(self.filepath)

class ImagePair(object):
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.same_person = (image1.person == image2.person)
        self.same_image = (image1.filepath == image2.filepath)

    def get_key(self, ignore_order):
        # if ignore_order then (image1,image2) == (image2,image1)
        # therefore, the key used to check if a pair already exists must then
        # catch both cases (A,B) and (B,A), i.e. it must be sorted to always
        # be (A,B)
        if ignore_order:
            #key = "$$$".join(sorted([fp1, fp2]))
            key = tuple(sorted([self.image1.filepath, self.image2.filepath]))
        else:
            #key = "$$$".join([fp1, fp2])
            key = tuple([self.image1.filepath, self.image2.filepath])
        return key

    def get_x(self):
        img1_x = misc.imread(self.image1.filepath)
        img2_x = misc.imread(self.image2.filepath)
        both = np.array([img1_x, img2_x]).astype(np.uint8)
        #both = np.concatenate([img1_x, img2_x]).astype(np.uint8)
        return both

def get_image_files(dataset_filepath, exclude_images=None):
    if not os.path.exists(dataset_filepath):
        raise Exception("Images filepath '%s' of the dataset seems to not exist." % (dataset_filepath,))

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

"""
def get_filepaths_by_name(dataset_filepath, seed=None):
    filepaths = get_filepaths(dataset_filepath, seed=seed)
    images = []
    for filepath in filepaths:
        name = filepath_to_person_name(name)
        images.append((name, filepath))

    images_by_person = defaultdict(list)
    for (img_fp, img_person_name) in images:
        images_by_person[img_person_name].append(img_fp)
"""

def get_image_pairs(dataset_filepath, nb_max, pairs_of_same_imgs=False, ignore_order=True, exclude_images=list(), seed=None, verbose=False):
    """Creates a list of tuples (filepath1, filepath1, (int)same_person)
    from images in a directory. 'filepath1' and 'filepath2' are the 
    full filepaths to images. 'same_person?' is 1 (int) if both images
    show the same person, otherwise it's 0.
    The images must be named according to lfwcrop_grey dataset (labeled
    faces in the wild, greayscaled and cropped). E.g.:
        Adam_Scott_0002.pgm
        Kalpana_Chawla_0002.pgm
    
    Args:
        images_filepath: Path to the directory which contains the images,
            without trailing slash.
        pairs_of_same_imgs: Whether pairs of images may be returned, where
            filepath1 == filepath2. Notice that this may return many
            pairs of same images as many people only have low amounts of
            images.
        ignore_order: Defines whether (filepath1, filepath2) shall be
            considered identical to (filepath2, filepath1). So if one
            of them is already added, the other one wont be added any more.
            Setting this to True will result in less possible but more
            diverse pairs of images.
        nb_max: Maximum number of images to return.
        not_in: Previous output of this function. No image will be picked
            that appears in that list.
    Returns:
        List of tuples (filepath1, filepath2, (int)same_person)
        Where (int)same_person is either 1 (same person in both images)
        or 0 (different person in both images).
    """
    if seed is not None:
        state = random.getstate() # used right before the return
        random.seed(seed)
    
    # Build set of filepaths to not use in image pairs (because they have
    # been used previously)
    exclude_images = set([img_pair.image1 for img_pair in exclude_images]
                         + [img_pair.image2 for img_pair in exclude_images])

    images = get_image_files(dataset_filepath, exclude_images=exclude_images)
    
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
    # ---
    
    # result
    pairs = []
    
    # counters
    nb_added = 0
    nb_same_p_same_img = 0
    nb_same_p_diff_img = 0
    nb_diff = 0
    
    # set that saves identifiers for pairs of images that have
    # already been added to the result.
    added = set() 
    
    # y = 1 (same person)
    while nb_added < nb_max // 2:
        person = random.choice(names_gte2)
        image1 = random.choice(images_by_person[person])
        if pairs_of_same_imgs:
            image2 = random.choice(images_by_person[person])
        else:
            image2 = random.choice([image for image in images_by_person[person] if image != image1])
        
        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)
        
        if key not in added:
            pairs.append(pair)
            nb_added += 1
            nb_same_p_same_img += 1 if pair.same_image else 0
            nb_same_p_diff_img += 1 if not pair.same_image else 0
            # log this pair as already added (dont add it a second time)
            added.add(key)
    
    # y = 0 (different person)
    while nb_added < nb_max:
        person1 = random.choice(names)
        person2 = random.choice([person for person in names if person != person1])
        
        # we dont have to check here whether the images are the same,
        # because they come from different persons
        image1 = random.choice(images_by_person[person1])
        image2 = random.choice(images_by_person[person2])
        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)
        
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
    
    if seed is not None:
        random.setstate(state) # state was set at the start of this function
    
    return pairs

def image_pairs_to_xy(image_pairs):
    """Converts tuples of (filepath1, filepath2, y) to tuples that may be
    used by the neural net.
    The structure of those tuples is
        (img1-fullscale + img2-fullscale,
         img1-smallscale + img2-smallscale,
         [1 if y==1 else 0, 1 if y==0 else 0])
    Where fullscale is 64x64 and smallscale is 32x32.
    Both images are numpy arrays.
    The last value in the tuple is either [1,0] (both images show the same
    person) or [0,1] (the images show different persons).
    Args:
        raw_examples_tuples: Tuples of the form (filepath1, filepath2, y)
            from get_filepaths_of_images().
    Returns:
        Tuples of the form
        (img1-fullscale+img2-fullscale,
         img1-smallscale+img2-smallscale,
         either [1,0] or [0,1]).
    """
    X = np.zeros((len(image_pairs), 2, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.uint8)
    y = np.zeros((len(image_pairs),), dtype=np.float32)
    
    for i, pair in enumerate(image_pairs):
        X[i] = pair.get_x()
        y[i] = 1 if pair.same_person else 0
    
    return X, y
