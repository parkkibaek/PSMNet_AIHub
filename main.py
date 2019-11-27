from __future__ import print_function
import argparse
import glob
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv

parser = argparse.ArgumentParser(description='PSMNet')

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batchsz', type=int ,default=12,
                    help='batch size')
parser.add_argument('--nworker', type=int ,default=12,
                    help='num_workers')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# random.seed(100)
# np.random.seed(100)
# # target directory construction for dataset
# target_location = "./data/sidewalk_dataset"
# sym_train_location = os.path.join(target_location+'/train')
# sym_test_location = os.path.join(target_location+'/test')
#
# mmcv.mkdir_or_exist(sym_train_location)
# mmcv.mkdir_or_exist(sym_test_location)
#
#
# image_folder_list = glob.glob(os.path.join(args.datapath, '1/ZED*'))
# image_folder_list.sort(reverse=False)
# assert len(image_folder_list) > 0
# random.shuffle(image_folder_list)
#
# num_test = int(len(image_folder_list) * 0.15)
# num_train = len(image_folder_list) - num_test
# train_folders = image_folder_list[:num_train]
# test_folders = image_folder_list[num_train:]
#
# # symlink image and annotations to train, val, test split
# symlink_images_to_target(train_folders, sym_train_location)
# symlink_images_to_target(test_folders, sym_test_location)
# # convert annotation file for each split
#
# import pdb
# pdb.set_trace()
# train_folder_list = glob.glob(os.path.join(sym_train_location, '*'))
# test_folder_list = glob.glob(os.path.join(sym_test_location, '*'))


all_left_disp = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/ZED*', '*disp16.png')))]
all_left_img = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/ZED*', '*left.png')))]
all_right_img = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/ZED*', '*right.png')))]

test_left_disp = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/test*', '*disp16.png')))]
test_left_img = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/test*', '*left.png')))]
test_right_img = [x for x in sorted(glob.glob(os.path.join(args.datapath,'*/test*', '*right.png')))]


TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= args.batchsz, shuffle= True, num_workers= args.nworker, drop_last=False)

TestImgLoader_fix = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= args.batchsz, shuffle= False, num_workers= args.nworker, drop_last=False)

TestImgLoader_orig = torch.utils.data.DataLoader(
         DA.myImageFloder2(test_left_img,test_right_img,test_left_disp, False),
         batch_size= args.batchsz, shuffle= False, num_workers= args.nworker, drop_last=False)



if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {} M'.format(np.float(sum([p.data.nelement() for p in model.parameters()]))/10**6))
print('Number of training image: {} K'.format(np.float(len(all_left_disp))/10**3))
print('Number of test image: {} K'.format(np.float(len(test_left_disp))/10**3))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        mask = disp_true < 192
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)

        # modified
        # output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]
        output = output3.data.cpu()

        if len(disp_true[mask])==0:
           loss = 0
           rate =0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error
           rate = (torch.sum(torch.abs(output[mask]-disp_true[mask]) < 3).float() / torch.sum(mask).float())*100

        return loss,output,rate

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    mmcv.mkdir_or_exist(args.savemodel)
    start_full_time = time.time()
    for epoch in range(1, args.epochs+1):

      total_train_loss = 0
      adjust_learning_rate(optimizer,epoch)

      ################################### TRAIN #########################################
      for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
          loss = train(imgL_crop,imgR_crop, disp_crop_L)
          total_train_loss += loss
      print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
      savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
      torch.save({
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'train_loss': total_train_loss/len(TrainImgLoader),}, savefilename)


      ################################### TEST #########################################
      # original PSMNet
      total_test_loss = 0
      total_test_rate = 0
      for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader_fix):

          test_loss, output, rate = test(imgL, imgR, disp_L)
          total_test_loss += test_loss
          total_test_rate += rate

      print('total test loss (reduced resolution) = %.3f' % (total_test_loss / len(TestImgLoader_fix)))
      print('total test rate (reduced resolution) = %.3f' % (total_test_rate / len(TestImgLoader_fix)))

      # test on original resolution
      total_test_loss = 0
      total_test_rate = 0
      for batch_idx, (imgL, imgR, disp_L, left) in enumerate(TestImgLoader_orig):

          test_loss, output, rate = test(imgL, imgR, disp_L)
          total_test_loss += test_loss
          total_test_rate += rate

      print('total test loss (original resolution) = %.3f' % (total_test_loss / len(TestImgLoader_orig)))
      print('total test rate (original resolution) = %.3f' % (total_test_rate / len(TestImgLoader_orig)))


def symlink_images_to_target(image_list, target_path):

    for source_image_path in image_list:
        img_filename = os.path.basename(source_image_path)
        target_img_path = os.path.join(target_path, img_filename)
        mmcv.symlink(source_image_path, target_img_path)

    print('images and annotations are copied to: {}'.format(target_path))

if __name__ == '__main__':
    main()

