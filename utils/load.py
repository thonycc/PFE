#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
from numpy import newaxis

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    if suffix == '.txt':
    	for id, pos in ids:
            im = resize_and_crop(Image.fromarray(np.loadtxt(dir + id + suffix)), scale=scale)
	    #pour passer en 3d
            im = im[:, :, newaxis]
	    #im = resize_and_crop(Image.fromarray(np.genfromtxt(dir + id + suffix)), scale=scale)
#	    yolo=np.genfromtxt(dir + id + suffix, delimiter=" -", invalid_raise=False)
#	    yolo2=np.isnan(yolo).all(axis=0)
#	    im = resize_and_crop(Image.fromarray(yolo[:,~yolo2]), scale=scale)

#default one            yield get_square(im, pos)
	    yield im
#test
    else:
    	for id, pos in ids:
	    im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
#	    im = resize_and_crop(Image.fromarray(np.genfromtxt(dir + id + suffix, delimiter=' -', invalid_raise=False).reshape(850,850), scale=scale)
            yield get_square(im, pos)
#test
def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.txt', scale)
   
    # need to transform from HWC to CHW
    #map(function_to_apply, list_of_inputs)
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    masks = to_cropped_imgs(ids, dir_mask, '.pgm', scale)
    
    return zip(imgs_normalized, masks)
#    return zip(imgs, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    #im = Image.open(dir_img + id + '.txt')
    im = Image.fromarray(np.loadtxt(dir + id + suffix))
    mask = Image.open(dir_mask + id + suffix)
    return np.array(mask), np.array(mask)
