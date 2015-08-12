# -*- coding: utf-8 -*-
import gzip
import cPickle as pickle

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
        
def save_model_weights(model, file_dir, file_name):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name
    
    model.save_weights(filepath)

def save_optimizer_state(optimizer, file_dir, file_name, overwrite=False):
    if not file_dir.endswith("/"):
        file_dir = file_dir + "/"
    filepath = file_dir + file_name
    
    state = optimizer.get_state()
    
    # Save weights from all layers to HDF5
    import h5py
    import os.path
    # if file exists and should not be overwritten
    if not overwrite and os.path.isfile(filepath):
        import sys
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
