import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from PIL import Image
import cv2
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, split_mask_image, mask_values, cut_masks, bin_mask

def train_net(net,
              epochs=1,
              batch_size=1,
              lr=0.001,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.8,
            #"background","class1","class2","class3"
              class_mark = [0,3,2,7]
              ):

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

 #   criterion = nn.BCELoss()
    criterion=torch.nn.CrossEntropyLoss()
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
           # print([j[0].size for j in b])
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            #imgs.shape = (batch_size, channels(3),height,width)
            true_masks = np.array([i[1] for i in b])
            #true_masks.shape = (batch_size,height,width)
            ##split a n_class mask into n masks
            #print(true_masks.shape)
            allmasks=true_masks
#put to 0 non class_mark values
            true_masks=cut_masks(true_masks,class_mark)
#et logique entre l'image et le mask
#possible in training but not prediction            imgs=imgs*binmasks
            #print(np.nonzero(true_masks))
#reclasse les classes non traités
#for bcl            true_masks = mask_values(true_masks, class_mark)
#for bcl            true_masks = split_mask_image(true_masks ,class_mark)
            #print('apres split')
            #print(true_mallmasks2=torch.from_numpy(allmasks)
            #print(true_masks)
            ##true_masks.shape = (batch_size,n,height,width)
            #print('truemask')
            #print(np.nonzero(true_masks))
            imgs= np.expand_dims(imgs, axis=1)
            #print('imgexpanded')
            #print(imgs.shape)
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            allmasks2=torch.from_numpy(allmasks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                allmasks2=allmasks2.cuda()

            masks_pred = net(imgs)
          #  print(masks_pred)
          #  print(np.nonzero(masks_pred))
           # print(masks_pred.)
           # print(true_masks.shape)            
#masks_pred.shape = (batch size, n, height,width)
            #true_masks_flat = true_masks.view(-1)
           # print(true_masks_flat)
            #masks_probs_flat = masks_pred.view(-1)
            #print(masks_probs_flat)
#            print(masks_pred.shape)
 #           print(true_masks.shape)
  #          print(allmasks2.shape)
            #loss = nn.functional.cross_entropy(masks_pred, allmasks2.long())
            #loss = nn.functional.cross_entropy(masks_pred, true_masks.long())
            #loss = criterion(masks_probs_flat, true_masks_flat)

            loss_class_weight=torch.ones(4).cuda()
            #loss_class_weight[0]=0.1
            #loss_class_weight[2]=0.8
            #loss_class_weight[1]=1
            #loss_class_weight[1]=0.6



            #loss2=F.cross_entropy(weight=loss_class_weight)
            #loss=loss2(masks_pred, allmasks2.long())
            loss = nn.functional.cross_entropy(masks_pred,true_masks.long(),weight=loss_class_weight)


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
       # allmasks=np.zeros((572,572), dtype=int)
       # for i in range(3):
          #  result = mask_to_image(mask[i])
          #  filename = str(i)+".png"
          #  result.save(filename)
        #     allmasks=allmasks+(i+7)*masks_pred[i]
        #print(allmasks[:,300])
       # print(allmasks.shape)
       # result=Image.fromarray(np.uint8(allmasks))
      #  filename='prediction{].pgm'.format(epoch + 1)
     #   result.save(filename)
##
#        print('Epoch finished ! Loss: {}'.format(epoch_loss))
#
        #if 1:
         #   val_dice = eval_net(net, val, gpu)
          #  print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.8, help='downscaling factor of the images')
##
    parser.add_option('-a', '--class', dest='class_mark',
                      default=[0,3,2,7], help='intensity in mask image')
##
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    # default one: net = UNet(n_channels=3, n_classes=1)
#dans autre cas comme on est en niveau de gris 1 seul channel?
    net = UNet(n_channels=1, n_classes=4)
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
