# -*- coding: utf-8 -*-
import gzip
import cPickle as pickle
import os
import os.path
import h5py
import sys

"""
def save_model(model, file_dir, file_name, file_extension, use_gzip=True):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name + file_extension
    
    if gzip:
        with gzip.open(filepath, "wb") as f:
            pickle.dump(model, f, -1)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(model, f, -1)

def load_model(file_dir, file_name, file_extension):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name + file_extension
    
    with gzip.open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def save_model_config(model, file_dir, file_name):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name
    
    with open(filepath, "wb") as f:
        f.write(str(model.get_config()))
"""

def save_model_weights(model, file_dir, file_name, overwrite=False):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name
    
    model.save_weights(filepath, overwrite=overwrite)

def save_optimizer_state(optimizer, file_dir, file_name, overwrite=False):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name
    
    state = optimizer.get_state()
    
    # Save weights from all layers to HDF5
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
        update_name = 'update_{}'.format(n)
        update_dset = group.create_dataset(update_name, update.shape, dtype=update.dtype)
        update_dset[:] = update

    f.flush()
    f.close()


def load_weights(model, save_weights_dir, previous_identifier):
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
