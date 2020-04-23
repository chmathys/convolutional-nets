import string
from tqdm import tqdm
import zipfile
import urllib.request as request
import gzip
import shutil
import os.path as path
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n) # will also set self.n = b * bsize

def get_emnist(dataset):
    """
    Loads one of the datasets in EMNIST, downloading the main zip file
    if it's not in the folder python is running in.
    
    It returns a tuple of the training and test set, in turn tuples
    containing the images and the labels (in categorical format):
    ((train_img, train_label), (test_img, test_label))

    Parameters
    ----------
    
    dataset: a str with the name of the dataset to load. can have the values of:
             - byclass: 814,255 characters. 62 unbalanced classes.
             - bymerge: 814,255 characters. 47 unbalanced classes.
             - balanced: 131,600 characters. 47 balanced classes.
             - letters: 145,600 characters. 26 balanced classes.
             - digits: 280,000 characters. 10 balanced classes.
             - mnist: 70,000 characters. 10 balanced classes.
    """
    
    emnist_url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    if not path.isfile('gzip.zip'):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=emnist_url.split('/')[-1]) as t: 
            # all optional kwargs
            request.urlretrieve(emnist_url, filename="gzip.zip", reporthook=t.update_to)
        with zipfile.ZipFile("gzip.zip", 'r') as zf:
            zf.extractall()
    
    with gzip.open(path.join("gzip", "emnist-"+dataset+"-train-images-idx3-ubyte.gz"), 'rb') as gf:
        with open("train-img", 'wb') as out_file:
            shutil.copyfileobj(gf, out_file)
    
    with gzip.open(path.join("gzip", "emnist-"+dataset+"-train-labels-idx1-ubyte.gz"), 'rb') as gf:
        with open("train-labels", 'wb') as out_file:
            shutil.copyfileobj(gf, out_file)
    
    with gzip.open(path.join("gzip", "emnist-"+dataset+"-test-images-idx3-ubyte.gz"), 'rb') as gf:
        with open("test-img", 'wb') as out_file:
            shutil.copyfileobj(gf, out_file)
    
    with gzip.open(path.join("gzip", "emnist-"+dataset+"-test-labels-idx1-ubyte.gz"), 'rb') as gf:
        with open("test-labels", 'wb') as out_file:
            shutil.copyfileobj(gf, out_file)

    train_i = np.memmap("train-img", mode='r', offset=16, order='C')
    train_i = np.array(train_i.reshape(-1, 28, 28, 1))
    train_i = np.swapaxes(train_i, 1, 2)
    train_i = train_i.astype(np.float32)/255.0
    train_l = np.memmap("train-labels", mode='r', offset=8)
    train_l = to_categorical(train_l-1)
    
    test_i = np.memmap("test-img", mode='r', offset=16, order='C')
    test_i = np.array(test_i.reshape(-1, 28, 28, 1))
    test_i = np.swapaxes(test_i, 1, 2)
    test_i = test_i.astype(np.float32)/255.0
    test_l = np.memmap("test-labels", mode='r', offset=8)
    test_l = to_categorical(test_l-1)
    
    return (train_i, train_l), (test_i, test_l)

def show_test_emnist(model, validation_data):
    letters = string.ascii_uppercase
    labels = np.argmax(model.predict(validation_data[0]), axis=1)
    ground_truth = np.argmax(validation_data[1], axis=1)
    accuracy = np.sum(labels != ground_truth)/labels.size
    num_errors = min(np.sum(labels != ground_truth), 20)
    errors = np.random.choice(np.nonzero(labels != ground_truth)[0], num_errors, replace=False)
    correct = np.random.choice(
        np.nonzero(labels == ground_truth)[0], 60 - num_errors, replace=False)
    stimuli = np.hstack((errors, correct))
    np.random.shuffle(stimuli)
    plt.style.use('grayscale')
    num_columns = 10
    num_rows = 6
    fig, axes = plt.subplots(num_rows, 10, figsize=(20, 3 * num_rows))
    for idx, ax in zip(stimuli, axes.ravel()):
        if idx in errors:
            c = 'r'
        else:
            c = 'k'
        ax.matshow(validation_data[0][idx, :, :, 0])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Prediction: '+letters[labels[idx]]+
                     ' \n Truth: '+letters[ground_truth[idx]], color=c)
    plt.suptitle("Model error rate: {:.2f}%".format(accuracy*100))
    plt.show()
    plt.style.use('seaborn')

def show_test(model, validation_data):
    labels = np.argmax(model.predict(validation_data[0]), axis=1)
    ground_truth = np.argmax(validation_data[1], axis=1)
    accuracy = np.sum(labels != ground_truth)/labels.size
    num_errors = min(np.sum(labels != ground_truth), 20)
    errors = np.random.choice(np.nonzero(labels != ground_truth)[0], num_errors, replace=False)
    correct = np.random.choice(
        np.nonzero(labels == ground_truth)[0], 60 - num_errors, replace=False)
    stimuli = np.hstack((errors, correct))
    np.random.shuffle(stimuli)
    plt.style.use('grayscale')
    num_columns = 10
    num_rows = 6
    fig, axes = plt.subplots(num_rows, 10, figsize=(20, 3 * num_rows))
    for idx, ax in zip(stimuli, axes.ravel()):
        if idx in errors:
            c = 'r'
        else:
            c = 'k'
        ax.matshow(validation_data[0][idx, :, :, 0])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Prediction: '+str(labels[idx])+
                     ' \n Truth: '+str(ground_truth[idx]), color=c)
    plt.suptitle("Model error rate: {:.2f}%".format(accuracy*100))
    plt.show()
    plt.style.use('seaborn')