import random
import numpy as np
import torch

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
    return np.transpose(img, axes=[2, 0, 1])
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
def resize_and_crop(pilimg, scale=0.7, final_height=572):
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
    # default img = img.crop((0, diff // 2, newW, newH - diff // 2))
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
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return (x+1024) / 2048 #255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

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
