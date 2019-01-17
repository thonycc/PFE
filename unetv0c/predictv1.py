#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms
from numpy import newaxis

def predict_img(net,
                full_img,
                scale_factor=0.8,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
#    img = normalize(img)
#
    print(img)
    img_height = img.shape[1]
    img_width = img.shape[0]

#
####    left_square, right_square = split_img_into_squares(img)
    right_square=img
    ####left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)
    #right_square=normalize(right_square)
    right_square= np.expand_dims(right_square, axis=0)



   #### X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)
    print(X_right)
    if use_gpu:
       #### X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        ####output_left = net(X_left)
        right_probs = net(X_right)
#default
#        left_probs = output_left.squeeze(0)
#        right_probs = output_right.squeeze(0)

#        tf = transforms.Compose(
#            [
#                transforms.ToPILImage(),
#                transforms.Resize(img_height),
#                transforms.ToTensor()
#            ]
#        )

        
#        left_probs = tf(left_probs.cpu())
#        right_probs = tf(right_probs.cpu())

## mutli segmentation add
        ####left_probs = F.sigmoid(output_left)
#dsmodel        right_probs = F.sigmoid(output_right)
        ####left_probs = F.upsample(left_probs, size=(img_height, img_height))
        right_probs = F.upsample(right_probs, size=(img_height, img_height))
##

        ####left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()
        print(right_mask_np)
#    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)
#
#    if use_dense_crf:
#        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    full_mask = np.zeros((right_mask_np.shape[0],img.shape[1],img.shape[0]))
    print(full_mask.shape)
####    full_mask = np.zeros((left_mask_np.shape[0],img.shape[1],img.shape[0]))
    for c in range(full_mask.shape[0]):
       #### full_mask[c] = merge_masks(left_mask_np[c], right_mask_np[c], img_width)
        full_mask[c]=right_mask_np[c]
        #if use_dense_crf:
            #full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask[c])
    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.8)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
#    out_files = get_output_filenames(args)

#    net = UNet(n_channels=3, n_classes=1)
## channel =3
    net = UNet(n_channels=1, n_classes=3)
##
    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
#lire txt
        img=Image.fromarray(np.loadtxt(fn))
 #       img=img[:, :, newaxis]
        
##        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)
##
        print(mask.shape)
##
        print(mask)
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:
#            out_fn = out_files[i]
#            result = mask_to_image(mask)
#            result.save(out_files[i])
##
            allmasks=np.zeros((572,572), dtype=int)
            for i in range(3):
                result = mask_to_image(mask[i])
                filename = str(i)+".png"
                result.save(filename)
                allmasks=allmasks+(i+1)*mask[i]
            print(allmasks[:,300])
            print(allmasks.shape)
            
            result=Image.fromarray(np.uint8(allmasks))
            filename="prediction.pgm"
            result.save(filename)
##

#            print("Mask saved to {}".format(out_files[i]))
