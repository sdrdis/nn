from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from keras import backend as K
from scipy.sparse.linalg.interface import _AdjointMatrixOperator
from os.path import join, isdir
import common


import keras.losses
import keras.metrics

import scipy.misc
import scipy.ndimage
import skimage.transform
from skimage.morphology import watershed
import argparse


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def masked_mean_squared_error(y_true, y_pred):
    mask = K.cast(Lambda(lambda x: x > -10, output_shape=(y_pred.shape[1], y_pred.shape[2], y_pred.shape[3]))(y_true),
                  'float32')
    diff = (y_true - y_pred) * mask
    squared_difference = K.sum(diff * diff)  # K.sum(((y_true - y_pred) * (y_true - y_pred)) * mask)
    squared_difference /= K.sum(mask)
    return squared_difference


def masked_mean_abs_error(y_true, y_pred):
    mask = K.cast(Lambda(lambda x: x > -10, output_shape=(y_pred.shape[1], y_pred.shape[2], y_pred.shape[3]))(y_true),
                  'float32')
    diff = (y_true - y_pred) * mask
    squared_difference = K.sum(K.abs(diff))  # K.sum(((y_true - y_pred) * (y_true - y_pred)) * mask)
    squared_difference /= K.sum(mask)
    return squared_difference


keras.metrics.dice_coef = dice_coef
keras.losses.dice_coef_loss = dice_coef_loss

keras.losses.masked_mean_squared_error = masked_mean_squared_error
keras.losses.masked_mean_abs_error = masked_mean_abs_error

K.set_image_data_format('channels_last')  # TF dimension ordering in this code




def get_base_weight(shape, unreliable_d=None):
    if (unreliable_d is None):
        unreliable_d = int(shape[0] / 5)
    base_weights = np.zeros(shape, dtype=bool)
    base_weights[1:-1, 1:-1] = True
    base_weights = scipy.ndimage.distance_transform_edt(base_weights)
    base_weights -= unreliable_d
    base_weights /= unreliable_d
    base_weights[base_weights < 0.000001] = 0.000001
    base_weights[base_weights > 1] = 1
    # scipy.misc.imsave('test/base_weights.png', base_weights)
    return base_weights


def predict(model, config_filepath, path):
    config = __import__(config_filepath, fromlist=[''])
    image_rows = config.input_shape[0]
    image_cols = config.input_shape[1]
    nb_dim_out = config.output_shape[2]
    patch_shape = (image_rows, image_cols)

    data_generator_module = __import__('data_generators.' + config.generator, fromlist=[''])

    X = scipy.misc.imread(path)

    X = config.image_preprocessing(X)



    #X = X[:,:,np.newaxis]

    X_pad = [(0, max(0, patch_shape[0] - X.shape[0])), (0, max(0, patch_shape[1] - X.shape[1]))]
    if (len(X.shape) == 3):
        X_pad.append((0, 0))
    X = np.lib.pad(X, X_pad,
                   'constant', constant_values=(0))



    patches = common.generate_patches(X.shape[:2], (image_rows, image_cols))

    imgs = []
    for patch in patches:
        ex_X = config.process_patch(X[patch[0]:patch[1], patch[2]:patch[3]], False)
        imgs.append(ex_X)

    imgs = np.array(imgs)
    imgs = data_generator_module.process_X(imgs)

    pred_np = model.predict(imgs, batch_size=32, verbose=1)

    base_weights = get_base_weight((image_rows, image_cols, nb_dim_out))

    all_pred_np = np.zeros((X.shape[0], X.shape[1], nb_dim_out))
    sum_np = np.zeros((X.shape[0], X.shape[1], nb_dim_out))
    for i in range(pred_np.shape[0]):
        all_pred_np[patches[i][0]:patches[i][1], patches[i][2]:patches[i][3]] += pred_np[i, :, :] * base_weights
        sum_np[patches[i][0]:patches[i][1], patches[i][2]:patches[i][3]] += base_weights

    all_pred_np /= sum_np

    return X, all_pred_np


add_image = False
# Kitti2015_000002_10
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on patches')
    parser.add_argument('config', type=str, help='Configuration')
    parser.add_argument('model', type=str, help='Model')
    parser.add_argument('input', type=str, help='Image')
    parser.add_argument('output', type=str, help='Output')
    parser.add_argument('--debug', type=bool, default=False, help='Save debug images in tmp folder')

    args = parser.parse_args()

    model = keras.models.load_model(args.model)

    im_np, pred_np = predict(model, args.config, args.input)

    np.savez_compressed(args.output, im_np=im_np, pred_np=pred_np)

    if (args.debug):
        tmp_folder = 'tmp'

        if (not isdir(tmp_folder)):
            os.makedirs(tmp_folder)

        scipy.misc.imsave(join(tmp_folder, 'im.png'), im_np)

        for i in range(pred_np.shape[2]):
            scipy.misc.imsave(join(tmp_folder, 'pred_'+str(i)+'.png'), pred_np[:, :, i])