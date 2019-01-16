import random
import numpy as np
import torch
import cv2
from numpy import newaxis

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
#    return np.transposer(img)
#	return img
    return np.transpose(img, axes=[0, 1])
#    return torch.Tensor.permute(img, axes=[2, 0, 1])
#test 
#    x = np.array(img)
#    x = torch.FloatTensor(x)
#    print(img.shape)
#    print(x.size())
#    print(img)
#    print(x)
#    x = x.permute(1, 0).contiguous()
#    x = x.view(1, 3, 256, 255)  # image (N x C x H x W)
#    return x
#fin test
def resize_and_crop(pilimg, scale=0.8, final_height=572):
    final_w=572
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)
#default
    if not final_height:
        diff = 0
    else:
        diff1 = newH - final_height
        diff2 = newW - final_w
#    diff = 6
    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff1 // 2, newW - diff2, newH - diff1 // 2))
#bricolage    #bricolage pour eviter le s572,573a cause de la floor division//
    img=pilimg.resize((572,572))
#fin bricolage 
   # default img = img.crop((0, diff // 2, newW, newH - diff // 2))
#    return np.array(img, dtype=np.float32)
##bricolage2pour3d
    #img2=np.array(img, dtype=np.float32)
    #img2 = img2[:, :, newaxis]
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
##
    trainset = dataset[:-n]
    valset = dataset[-n:]
##
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return (x) / 1024 #255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

#return as np.float32
    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
##


def split_mask_image2(mask_image,class_mark):
    batch_size = mask_image.shape[0]
    n_classes = len(class_mark)
    split_mask = np.zeros((batch_size,n_classes - 1,mask_image.shape[1],mask_image.shape[2]),np.float32)
    for bs in range (batch_size):
        #except 0
        for i in range(1,n_classes):
            temp_mask = mask_image[bs].copy()
            temp_mask[temp_mask!=class_mark[i]]=0
            temp_mask = temp_mask/class_mark[i]
            split_mask[bs,i-1,:,:] = temp_mask
    return split_mask



def split_mask_image(mask_image,class_mark):
    batch_size = mask_image.shape[0]
    n_classes = len(class_mark)
    split_mask = np.zeros((batch_size,n_classes,mask_image.shape[1],mask_image.shape[2]),np.float32)

    for bs in range (batch_size):
        #except 0
        for i in range(0,n_classes-1):
            temp_mask = mask_image[bs].copy()
            if i==0:
                temp_mask[temp_mask==class_mark[i]]=-1
                temp_mask[temp_mask!=-1]=0
                temp_mask[temp_mask==-1]=1
                split_mask[bs,i,:,:] = temp_mask

            else:
                temp_mask[temp_mask!=class_mark[i]]=0
                temp_mask = temp_mask/class_mark[i]
                split_mask[bs,i,:,:] = temp_mask
    return split_mask
##
def cut_masks(mask_image,class_mark):
    batch_size = mask_image.shape[0]
    n_classes = len(class_mark)
    cutorder_masks = np.zeros((batch_size,n_classes,mask_image.shape[1],mask_image.shape[2]),np.float32)
    cut_masks = np.zeros((batch_size,mask_image.shape[1],mask_image.shape[2]),np.float32)
    fcut_masks = np.zeros((batch_size,mask_image.shape[1],mask_image.shape[2]),np.float32)

    for bs in range (batch_size):

        #except 0
        temp_mask0 = mask_image[bs].copy()
        for i in [1,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]:
          #  temp_mask = mask_image[bs].copy()
            temp_mask0[temp_mask0==i]=3
            cut_masks[bs,:,:] = temp_mask0
        for i in range(1,n_classes):
            temp_mask = cut_masks[bs].copy()
            temp_mask[temp_mask!=class_mark[i]]=0
            temp_mask = temp_mask/class_mark[i]
#            temp_mask[temp_mask==class_mark[i]]=i
            cutorder_masks[bs,i,:,:] = temp_mask
            fcut_masks[bs,:,:]=fcut_masks[bs,:,:]+i*temp_mask    
#        fcut_masks0 = np.ones((batch_size,mask_image.shape[1],mask_image.shape[2]),np.float32)
 #       binmasks=bin_mask(mask_image)
#        0_mask=fcut_masks 

    return fcut_masks

#change non class_mark value
#to try, I think it doesnt do as expected
def mask_values(mask_image,class_mark):
    batch_size = mask_image.shape[0]
    n_classes = len(class_mark)
    masks_reduce = np.zeros((batch_size,mask_image.shape[1],mask_image.shape[2]),np.float32)
    for bs in range (batch_size):
        #except 0
        temp_mask = mask_image[bs].copy()
        temp_mask[temp_mask.all() not in class_mark]=2

        masks_reduce[bs,:,:] = temp_mask
    return masks_reduce

#binarize un mask en mettant à 1 toutes les valeurs non nulles
def bin_mask(mask_image):
    batch_size = mask_image.shape[0]
    bin_mask = np.zeros((batch_size,mask_image.shape[1],mask_image.shape[2]),np.float32)
    for bs in range (batch_size):
        temp_mask = mask_image[bs].copy()
        temp_mask[temp_mask!=0]=1
        bin_mask[bs,:,:] = temp_mask
    return bin_mask

#ctransform class values to 1 2 3 4 5 6  in order

