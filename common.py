import numpy as np
from os.path import join, isfile, isdir
from os import listdir
import json
from skimage.draw import polygon
import keras.utils
import scipy.misc
import random
from skimage.morphology import disk, greyreconstruct


def generate_patches(shape, patch_shape, padding=None):
    if (padding is None):
        padding = int(patch_shape[0] / 2)
    extracts = []
    y_i = 0
    while(True):
        fy = y_i * (patch_shape[0] - padding)
        ty = fy + patch_shape[0]
        is_out = ty >= shape[0]
        if (is_out):
            ty = shape[0]
            fy = ty - patch_shape[0]
        x_i = 0
        while (True):
            fx = x_i * (patch_shape[1] - padding)
            tx = fx + patch_shape[1]

            must_break = tx >= shape[1]
            if (must_break):
                tx = shape[1]
                fx = tx - patch_shape[1]
            extracts.append([fy, ty, fx, tx])
            x_i += 1
            if (must_break):
                break
        y_i += 1
        if (is_out):
            break
    return extracts

def _std_im(im_np, nb=4, epsilon=0.001):
    sel_im_np = im_np[:, :]
    sel_im_np -= np.mean(sel_im_np)
    sel_im_np /= (np.std(sel_im_np) + epsilon) * nb
    sel_im_np[sel_im_np < -1] = -1
    sel_im_np[sel_im_np > 1] = 1
    return sel_im_np

def standardize_im(im_np, nb=4):
    im_np = im_np.astype(float)
    if (len(im_np.shape) == 3):
        for i in range(im_np.shape[2]):
            im_np[:,:,i] = _std_im(im_np[:,:,i], nb)
    else:
        im_np = _std_im(im_np, nb)
    im_np += 1
    im_np *= 127
    return im_np.astype('uint8')

def load_gt(json_path, img_filename, shape):
    gt_np = np.zeros(shape, dtype=bool)
    with open(join(json_path, img_filename[:-4] + '_polygon.json'), 'r') as f:
        json_info = json.load(f)
        json_info = json_info['Polygon'][0]

    r = []
    c = []
    for el in json_info:
        c.append(el[0])
        r.append(el[1])

    rr, cc = polygon(r, c)
    gt_np[rr,cc] = 1

    return gt_np

def get_shuffled_sublist(lst, max_nb = None):
    lst_np = np.array(lst)
    if (max_nb is None):
        max_nb = lst_np.shape[0]
    idx = np.arange(lst_np.shape[0])
    np.random.shuffle(idx)
    idx = idx[:max_nb]
    lst_np = lst_np[idx]
    return lst_np.tolist()

def get_rejected_filenames():
    filepath = 'DOTA_files_rejected.txt'
    rejected = {}
    with open(filepath, 'r') as f:
        str = f.read()
        for line in str.split('\n'):
            line_data = line.split(' ')
            folder = line_data[0]
            filename = line_data[2]
            if (folder not in rejected):
                rejected[folder] = set([])
            rejected[folder].add(filename)
    return rejected


def h_minima(image, h, selem=None):
    shifted_img = image + h

    rec_img = greyreconstruct.reconstruction(shifted_img, image,
                                             method='erosion', selem=selem) #grey_reconstruction_erosion(shifted_img, image)

    residue_img = rec_img - image
    return residue_img > 0

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#def get_instances(conf_np, seed_np):

