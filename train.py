from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import keras.models
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, multiply, Lambda, \
    BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras import backend as K
from keras import losses

from os.path import join, isfile, isdir
import argparse

parser = argparse.ArgumentParser(description='Train on patches')
parser.add_argument('config_filepath', type=str, help='Configuration file')
parser.add_argument('--from_model', type=str, help='Starting model')

args = parser.parse_args()

models_path = 'models'

config = __import__(args.config_filepath, fromlist=[''])

K.set_image_data_format('channels_last')  # TF dimension ordering in this code



def get_model_path(models_path, model_name, model_i):
    dir_path = join(models_path, model_name)
    if (not(isdir(dir_path))):
        os.makedirs(dir_path)
    return join(dir_path, str(model_i) + '.h5')

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = optimizer.lr.get_value()
        print('\nLR: {:.6f}\n'.format(lr))

def train_and_predict(nb_epochs=500 * 7):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    #np.random.seed(1988)

    if (isinstance(config.generator, str)):
        data_generator_module = __import__('data_generators.' + config.generator, fromlist=[''])
        DataGenerator = data_generator_module.DataGenerator
    else:
        DataGenerator = config.generator()

    train_path = join(config.path, 'training')  # '../robust_stereo/patches_mccnn_diffused/' #'C:\\test\\robust_stereo\\patches_mccnn_diffused\\'
    validation_path = join(config.path, 'validation')

    training_generator = DataGenerator(train_path, DataGenerator.get_ids(train_path), batch_size=config.batch_size, process_patch=config.process_patch)
    validation_generator = DataGenerator(validation_path, DataGenerator.get_ids(validation_path), batch_size=config.batch_size, process_patch=config.process_patch)

    # imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    if (args.from_model is None):
        model = config.get_model() # get_unet()
    else:
        model = keras.models.load_model(args.from_model)
    model_name = args.config_filepath.split('.')[-1]
    model_i = 1
    while (isfile(get_model_path(models_path, model_name, model_i))):
        model_i += 1

    model_path = get_model_path(models_path, model_name, model_i)

    print ('Saving checkpoint on:', model_path)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=nb_epochs,
                        callbacks=[model_checkpoint, reduce_lr],
                        verbose=1)

    '''
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=500, verbose=1, shuffle=True,
              validation_split=0.05,
              callbacks=[model_checkpoint])
    '''
    # np.savez_compressed('model_aux_data.npz', mean=mean, std=std)


if __name__ == '__main__':
    train_and_predict()
