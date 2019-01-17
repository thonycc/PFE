#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
from numpy import newaxis
import cv2 

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
#deplace dans utils            im = im[:, :, newaxis]
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
### rajouter une dimension comme pour les images txt?
#	    im = resize_and_crop(Image.fromarray(np.genfromtxt(dir + id + suffix, delimiter=' -', invalid_raise=False).reshape(850,850), scale=scale)
#default            yield get_square(im, pos)
            yield im
#test
def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.txt', scale)
   
    # need to transform from HWC to CHW
    #map(function_to_apply, list_of_inputs)
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    masks = to_cropped_imgs(ids, dir_mask, '.pgm', scale)

#zip: Make an iterator that aggregates elements from each of the iterables.

#Returns an iterator of tuples, where the i-th tuple contains the i-th #element from each of the argument sequences or iterables. The iterator #stops when the shortest input iterable is exhausted. With a single #iterable argument, it returns an iterator of 1-tuples. With no #arguments, it returns an empty iterator. Equivalent to:    
    return zip(imgs_normalized, masks)
#    return zip(imgs_switched, masks)
#    return zip(imgs, masks)


#def get_full_img_and_mask(id, dir_img, dir_mask):
#    #im = Image.open(dir_img + id + '.txt')
#    im = Image.fromarray(np.loadtxt(dir + id + suffix))
#    mask = Image.open(dir_mask + id + suffix)
#    return np.array(mask), np.array(mask)
##
def get_full_img_and_mask(ids, dir_img, dir_mask,suffix_img, suffix_mask, target_size):
    #train as full size image
    list1 = []
    for id, pos in ids:
        im_p = Image.open(dir_img + id + suffix_img)
        mask_p = Image.open(dir_mask + id + suffix_mask)
        if(target_size[0]!=0):
            # PIL.resize(width,height)
            im_p = im_p.resize((target_size[1], target_size[0]))
            mask_p = mask_p.resize((target_size[1], target_size[0]))
        mask = np.array(mask_p, dtype=np.float32)
        im = np.array(im_p, dtype=np.float32)
        im = hwc_to_chw(im)
        list1.append((im, mask))
    return list1
##
