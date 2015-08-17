# -*- coding: utf-8 -*-
import gzip
import cPickle as pickle
import os
import os.path
import h5py
import sys

def save_model_weights(model, file_dir, file_name, overwrite=False):
    """Save the weights of a model/neural net.

    Args:
        model: The neural net.
        file_dir: Directory in which to create the file.
        file_name: Name of the file to write to.
        overwrite: Whether to overwrite an existing file. If set to False,
            the program will stop and ask whether to overwrite the content.
    """
    filepath = os.path.join(file_dir, file_name)
    model.save_weights(filepath, overwrite=overwrite)

def save_optimizer_state(optimizer, file_dir, file_name, overwrite=False):
    """Save the state of a neural net optimizer, e.g. Adagrad, SGD, ...

    Only really tested with Adagrad.

    Args:
        model: The neural net.
        file_dir: Directory in which to create the file.
        file_name: Name of the file to write to.
        overwrite: Whether to overwrite an existing file. If set to False,
            the program will stop and ask whether to overwrite the content.
    """
    
    filepath = os.path.join(file_dir, file_name)

    state = optimizer.get_state()

    # Note:
    #  the following content is mostly copied from keras' save_weights function

    # Save parameters from all layers to HDF5
    # if file exists and should not be overwritten
    if not overwrite and os.path.isfile(filepath):
        get_input = input
        if sys.version_info[:2] <= (2, 7):
            get_input = raw_input
        overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
        while overwrite not in ['y', 'n']:
            overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
        if overwrite == 'n':
            return
        print('[TIP] Next time specify overwrite=True in save_weights!')

    f = h5py.File(filepath, 'w')
    group = f.create_group('updates')
    group.attrs['nb_updates'] = len(state)
    for n, update in enumerate(state):
        is_scalar = True if len(update.shape) == 0 else False
        shape = (1,) if is_scalar else update.shape
        #uplen = len(np.atleast_1d(update)) # catch case where update-array is 1d-scalar
        
        update_name = 'update_{}'.format(n)
        #update_dset = group.create_dataset(update_name, update.shape, dtype=update.dtype)
        update_dset = group.create_dataset(update_name, shape, dtype=update.dtype)
        
        #print(update, type(update), update.shape, shape) #, len(update))
        if not is_scalar > 1:
            update_dset[:] = update
        else:
            update_dset[0] = update[0]

    f.flush()
    f.close()


def load_weights(model, save_weights_dir, previous_identifier):
    """Load the weights of an older experiment into a model.
    
    This function searches for files called "<previous_identifier>.at1234.weights"
    or "<previous_identifier>.last.weights" (wehre at1234 represents epoch 1234).
    If a *.last file was found, that one will be used. Otherwise the weights file
    with the highest epoch number will be used.
    
    The new and the old model must have identical architecture/layers.
    
    Args:
        model: The model for which to load the weights. The current weights
            will be overwritten.
        save_weights_dir: The directory in which weights are saved.
        previous_identifier: Identifier of the old experiment.
    Returns:
        Either tuple (bool success, int epoch)
            or tuple (bool success, string "last"),
        where "success" indicates whether a weights file was found
        and "epoch" represents the epoch of that weights file (e.g. 1234 in *.at1234)
        and "last" represents a *.last file.
    """
    filenames = [f for f in os.listdir(save_weights_dir) if os.path.isfile(os.path.join(save_weights_dir, f))]
    filenames = [f for f in filenames if f.startswith(previous_identifier + ".") and f.endswith(".weights")]
    if len(filenames) == 0:
        return (False, -1)
    else:
        filenames_last = [f for f in filenames if f.endswith(".last.weights")]
        if len(filenames_last) >= 2:
            raise Exception("Ambiguous weight files for model, multiple files match description.")
        if len(filenames_last) == 1:
            weights_filepath = os.path.join(save_weights_dir, filenames_last[0])
            load_weights_seq(model, weights_filepath)
            return (True, "last")
        else:
            # If we have a filename, e.g. "model1.at500.weights", we split it at
            # every "." so that we get ["model1", "at500", "weights"], we then
            # pick the 2nd entry ("at500") and convert the digits to an
            # integer (500). We sort the list of ints in reverse to have the
            # highest value at the first position (e.g. [500, 400, 300, ...]).
            epochs = sorted([int(re.sub("[^0-9]", "", f.split(".")[1])) for f in filenames], reverse=True)
            fname = "{}.at{}.weights".format(previous_identifier, epochs[0])
            weights_filepath = os.path.join(save_weights_dir, fname)
            #model.load_weights(weights_filepath)
            load_weights_seq(model, weights_filepath)
            return (True, epochs[0])

def load_weights_seq(seq, filepath):
    """Loads the weights from an exactly specified weights file into a model.
    
    This function is identical to Kera's load_weights function, but checks first
    if the file exists and raises an error if that is not the case.
    
    In contrast to the load_weights function above, this one expects the full
    path to the weights file and does not search on its own for a well fitting one.
    
    Args:
        seq: The model for which to load the weights. The current weights
            will be overwritten.
        filepath: Full path to the weights file.
    """
    # Loads weights from HDF5 file
    if not os.path.isfile(filepath):
        raise Exception("Weight file '%s' does not exist." % (filepath,))
    else:
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            seq.layers[k].set_weights(weights)
        f.close()

def load_optimizer_state(optimizer, save_optimizer_state_dir, previous_identifier):
    """Loads the state of an optimizer from a previous experiment.
    
    This function works similar to the load_weights function and searches for
    "<identifier>.at1234.optstate" or "<identifier>.last.optstate" files in
    the provided directory.
    
    This function was only really tested with Adagrad and seemed to cause errors
    with Adam.
    
    Args:
        optimizer: The optimizer for which to load the state. The current state
            will be overwritten.
        save_optimizer_state_dir: The directory in which optimizer states are
            saved.
        previous_identifier: The identifier of the old experiment from which
            to load the optimizer state.
    Returns:
        Either tuple (bool success, int epoch)
            or tuple (bool success, string "last"),
        where "success" indicates whether a optimizer state file was found
        and "epoch" represents the epoch of that file (e.g. 1234 in *.at1234)
        and "last" represents a *.last file.
    """
    odir = save_optimizer_state_dir
    filenames = [f for f in os.listdir(odir) if os.path.isfile(os.path.join(odir, f))]
    filenames = [f for f in filenames if f.startswith(previous_identifier + ".") and f.endswith(".optstate")]
    if len(filenames) == 0:
        return (False, -1)
    else:
        filenames_last = [f for f in filenames if f.endswith(".last.optstate")]
        if len(filenames_last) >= 2:
            raise Exception("Ambiguous optimizer state files for model, multiple files match description.")
        if len(filenames_last) == 1:
            optstate_filepath = os.path.join(odir, filenames_last[0])
            optstate_epoch = "last"
        else:
            # If we have a filename, e.g. "model1.at500.weights", we split it at
            # every "." so that we get ["model1", "at500", "weights"], we then
            # pick the 2nd entry ("at500") and convert the digits to an
            # integer (500). We sort the list of ints in reverse to have the
            # highest value at the first position (e.g. [500, 400, 300, ...]).
            epochs = sorted([int(re.sub("[^0-9]", "", f.split(".")[1])) for f in filenames], reverse=True)
            fname = "{}.at{}.optstate".format(previous_identifier, epochs[0])
            optstate_filepath = os.path.join(odir, fname)
            optstate_epoch = epochs[0]
            
        # Loads state from HDF5 file
        if not os.path.isfile(optstate_filepath):
            raise Exception("Optimizer state file '%s' does not exist." % (optstate_filepath,))
        else:
            f = h5py.File(optstate_filepath)
            g = f['updates']
            nb_updates = g.attrs["nb_updates"]
            updates = [g['update_{}'.format(p)] for p in range(nb_updates)]
            optimizer.set_state(updates)
            f.close()
            
        return (True, optstate_epoch)
