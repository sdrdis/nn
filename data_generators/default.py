import keras
import numpy as np
from os.path import join, isfile
import scipy.misc
from os import listdir

def process_X(X):
    X = X.astype(float)
    X -= 128
    X /= 128

    if (len(X.shape) == 3):
        X = X[:, :, :, np.newaxis]

    return X

def process_Xy(X, y):
    X = process_X(X)
    if (len(y.shape) == 1):
        y = y[:, np.newaxis]
    if (len(y.shape) == 3):
        y = y[:, :, :, np.newaxis]
    return X, y


class DataGenerator(keras.utils.Sequence):
    path = None#'patches_fast/'

    'Generates data for Keras'
    def __init__(self, path, list_IDs, batch_size=24, shuffle=True, process_patch=None):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.process_patch = process_patch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        list_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, comp_X, y = self.__data_generation(list_IDs_temp)

        if (comp_X.shape[0] == 0 or comp_X[0] is None):
            return X, y
        else:
            return [X, comp_X], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.list_IDs.copy() #np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp, randomize=True):
        path = self.path
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        y = []
        comp_X = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im_np = scipy.misc.imread(join(path, str(ID) + '_X.png'))
            if (self.process_patch is not None):
                im_np = self.process_patch(im_np, True)
            y_np = np.load(join(path, str(ID) + '_y.npz'))['y'] #.astype(float)

            comp_X_filepath = join(path, str(ID) + '_comp_X.npz')
            comp_X_np = None
            if (isfile(comp_X_filepath)):
                comp_X_np = np.load(comp_X_filepath)['complementary_X'].tolist()
                comp_X_np = np.hstack(list(comp_X_np.values()))
            X.append(im_np)
            comp_X.append(comp_X_np)
            y.append(y_np)

        X = np.array(X)
        comp_X = np.array(comp_X)
        y = np.array(y)
        X, y = process_Xy(X, y)
        return X, comp_X, y

    @staticmethod
    def get_ids(path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(files)
        ids = []
        for file in files:
            if (file.endswith('_X.png')):
                ids.append(file[:-6])
        return np.array(ids)