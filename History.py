import csv

class History(object):
    """A simple object to save the training history of an experiment to.
    History included: Epochs, Loss (training set), Loss (validation set),
    Accuracy (training set), Accuracy (validation set).
    
    Can easily be saved to a csv file.
    """
    def __init__(self):
        """Initialize the history."""
        self.epochs = []
        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []

    def add(self, epoch, loss_train=None, loss_val=None, acc_train=None, acc_val=None):
        """Add an entry (row) to the history.
        
        Should work in principle without providing all values, but nevertheless
        you should provide all. Used named attributes here mostly for clarity when
        calling the function, so that values don't get mixed up.
        
        Args:
            epoch: The epoch of the other values (i.e. of the row).
            loss_train: The loss value of the training set of the epoch.
            loss_val: The loss value of the validation set of the epoch.
            acc_train: The accuracy value of the training set of the epoch.
            acc_val: The accuracy value of the validation set of the epoch.
        """
        self.epochs.append(epoch)
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)
        self.acc_train.append(acc_train)
        self.acc_val.append(acc_val)

    def add_all(self, start_epoch, loss_train, loss_val, acc_train, acc_val):
        """Add lists of values to the history.
        
        All lists must have equal lengths.
        
        Args:
            start_epoch: Epoch of the first value.
            loss_train: List of the values of the loss of the training set.
            loss_val: List of the values of the loss of the validation set.
            acc_train: List of the values of the accuracy of the training set.
            acc_val: List of the values of the accuracy of the validation set.
        """
        last_epoch = start_epoch + len(loss_train)
        for epoch, lt, lv, at, av in zip(range(start_epoch, last_epoch+1), loss_train, loss_val, acc_train, acc_val):
            self.add(epoch, loss_train=lt, loss_val=lv, acc_train=at, acc_val=av)

    def save_to_filepath(self, csv_filepath):
        """Saves the contents of the history to a csv file.
        
        Args:
            csv_filepath: Full path to the file to write to. All content in the
                file will be completely overwritten.
        """
        with open(csv_filepath, "w") as fp:
            csvw = csv.writer(fp, delimiter=",")
            # header row
            rows = [["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]]
            
            rows.extend(zip(self.epochs, self.loss_train, self.loss_val, self.acc_train, self.acc_val))
            csvw.writerows(rows)

    def load_from_file(self, csv_filepath, last_epoch=None):
        """Loads the content of the history from a csv file.
        
        It is assumed that the csv file has the same structure as the one
        created by save_to_filepath().
        
        Args:
            csv_filepath: Full path to the file to read.
            last_epoch: The epoch until which to read the content (including).
                E.g. last_epoch=10 will read the rows for epoch 1, 2, 3, ... and 10.
                If set to "last" or None then all epochs will be read.
                Default is None (read all).
        """
        # load previous loss/acc values per epoch from csv file
        csv_lines = open(csv_filepath, "r").readlines()
        csv_lines = csv_lines[1:] # no header
        csv_cells = [line.strip().split(",") for line in csv_lines]
        epochs = [int(cells[0]) for cells in csv_cells]
        stats_loss_train = [float(cells[1]) for cells in csv_cells]
        stats_loss_val = [float(cells[2]) for cells in csv_cells]
        stats_acc_train = [float(cells[3]) for cells in csv_cells]
        stats_acc_val = [float(cells[4]) for cells in csv_cells]
        
        if last_epoch is not None and last_epoch is not "last":
            epochs = epochs[0:last_epoch+1]
            stats_loss_train = stats_loss_train[0:last_epoch+1]
            stats_loss_val = stats_loss_val[0:last_epoch+1]
            stats_acc_train = stats_acc_train[0:last_epoch+1]
            stats_acc_val = stats_acc_val[0:last_epoch+1]
        
        self.epochs = epochs
        self.loss_train = stats_loss_train
        self.loss_val = stats_loss_val
        self.acc_train = stats_acc_train
        self.acc_val = stats_acc_val
