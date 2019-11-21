import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import preprocess 
import listflowfile as lt
import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    try:
        data = Image.open(path).convert('RGB')
        if np.array(data).ndim != 3:
            import pdb
            pdb.set_trace()
        return data
    except IOError:
        print(path)
    except SyntaxError:
        print(path)

        return None


# def disparity_loader(path):
#     return rp.readPFM(path)

def disparity_loader(path):
    try:
        data = Image.open(path)
        return data
    except IOError:
        print(path)
    except SyntaxError:
        print(path)
        return None



class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
    def __getitem__(self, index):

        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        assert left.split('/')[-2] == right.split('/')[-2] , "folder is not matched {}".format(left)
        assert left.split('/')[-1][:-9] == right.split('/')[-1][:-10], "L,R is not matched {},{},{},{}".format(left.split('/')[-1],right.split('/')[-1],left,index) 
        assert left.split('/')[-1][:-9] == disp_L.split('/')[-1][:-11], "L,disp is not matched {},{},{},{}".format(left.split('/')[-1][:-9],disp_L.split('/')[-1][:-11],left,index) 

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        try:
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        except TypeError:
            print("data is wrong. data path is {}".format(disp_L))

        w, h = left_img.size
        left_img = left_img.crop((0, 244, w, 244+592))
        try:
            right_img = right_img.crop((0, 244, w, 244+592))
        except AttributeError:
            print(right)


        dataL = dataL[244:-244,:]/100.0*1920/1248 # Ground Truth output coding 
        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size
           left_img = left_img.crop((w-960, h-544, w, h))
           right_img = right_img.crop((w-960, h-544, w, h))
           processed = preprocess.get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           dataL = dataL[-544:,-960:]
           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


class myImageFloder2(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):

        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        assert left.split('/')[-2] == right.split('/')[-2] , "L,R is not matched {}".format(left)
        assert left.split('/')[-1][:-9] == right.split('/')[-1][:-10], "L,R is not matched {},{},{},{}".format(left.split('/')[-1],right.split('/')[-1],left,index) 
        assert left.split('/')[-1][:-9] == disp_L.split('/')[-1][:-11], "L,disp is not matched {},{},{},{}".format(left.split('/')[-1],right.split('/')[-1],left,index) 

        left_img = self.loader(left)
        right_img = self.loader(right)
        # dataL, scaleL = self.dploader(disp_L)
        dataL = self.dploader(disp_L)
        try:
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        except TypeError:
            print(dataL)
            print(disp_L)

        w, h = left_img.size
        left_img = left_img.crop((0, 244, w, 244 + 592))
        right_img = right_img.crop((0, 244, w, 244 + 592))
        dataL = dataL[244:-244, :] / 100.0  # Ground Truth output coding

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            left_img = left_img
            right_img = right_img
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, left

    def __len__(self):
        return len(self.left)


