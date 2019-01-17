import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
from eval import eval_net
from unet2 import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, split_mask_image, cut_masks

def train_net(net,
              epochs=1,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.9,
            #"background","class1","class2","class3"
              class_mark = [0,2,7,3],
              padding=True,
              ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_img = '/space/homes/student02/pfe/pfe/datapfe/images/'
    dir_mask = '/space/homes/student02/pfe/pfe/datapfe/masks/'
    dir_checkpoint = '/space/homes/student02/pfe/pfe/unetv0c/checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}

    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

   # criterion = F.cross_entropy()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
	    
           # imgs = np.array(np.asarray([i[0] for i in b]), dtype=object) #.astype(np.float32)
	    #default one: imgs = np.array([i[0] for i in b]).astype(np.float32)

            imgs = np.array([i[0] for i in b]).astype(np.float32)
            #imgs.shape = (batch_size, channels(3),height,width)
            true_masks = np.array([i[1] for i in b])
            #true_masks.shape = (batch_size,height,width)
            ##split a n_class mask into n masks
            allmasks=true_masks
            allmasks=cut_masks(allmasks,class_mark)
            allmasks2=torch.from_numpy(allmasks)
            true_masks = split_mask_image(true_masks ,class_mark)
            
            ##true_masks.shape = (batch_size,n,height,width)
            imgs= np.expand_dims(imgs, axis=1)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                allmasks2=allmasks2.cuda()

            masks_pred = net(imgs)
          
#masks_pred.shape = (batch size, n, height,width)
            masks_probs_flat = masks_pred.view(-1)
            #print(masks_probs_flat)
            true_masks_flat = true_masks.view(-1)
            pourresize=torch.nn.Upsample((572,572))
            masks_pred=pourresize(masks_pred)
            print(masks_pred.shape)
#classweight 4=n_classes
            masks_predsig=F.sigmoid(masks_pred)
            loss_class_weight=torch.ones(4).cuda()
            loss_class_weight[0]=0
            #loss2=F.cross_entropy(weight=loss_class_weight)
            #loss=loss2(masks_pred, allmasks2.long())
            loss = F.cross_entropy(masks_predsig,allmasks2.long(),weight=loss_class_weight)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
##
#            print('loss: {0:.6f} '.format(loss.item()))
##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / batch_size))
        #pour sauvegarder une prediction a chaque epoch
        #result=Image.fromarray(np.uint8(allmasks))
        #filename='prediction{].pgm'.format(epoch + 1)
        #result.save(filename)
##
#        print('Epoch finished ! Loss: {}'.format(epoch_loss))
#
      #  if 1:
       #     val_dice = eval_net(net, val, gpu)
        #    print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + '2CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.9, help='downscaling factor of the images')
##
    parser.add_option('-a', '--class', dest='class_mark',
                      default=[0,5,7,3], help='intensity in mask image')
##
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    # default one: net = UNet(n_channels=3, n_classes=1)
#dans autre cas comme on est en niveau de gris 1 seul channel?
    #net = UNet(n_channels=1, n_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_classes=4, padding=True, up_mode='upsample').to(device)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  class_mark=args.class_mark)
##last parameter
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
