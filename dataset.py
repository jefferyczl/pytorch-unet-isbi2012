import torch.utils.data as data
import torch
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import random
import u_transform as ut
class ISBI2012Dataset(data.Dataset):
    def __init__(
        self,
        root,
        image_set='train',
        transform = None ,
        augment = False):
        self.root = root
        self.transform = transform
        self.image_set = image_set
        isbi_root = os.path.join(self.root,'isbi2012')
        self.image_dir = os.path.join(isbi_root,image_set,'imgs')
        self.augment = augment and image_set == 'train'
        if not os.path.isdir(isbi_root):
            raise RuntimeError('Dataset not found')
        self.mask_dir = os.path.join(isbi_root,image_set,'labels')
        self.names = os.listdir(self.image_dir)
        if self.augment:
            self.augment_trans = transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=2,#roate scale
                        translate = (0.05,0.05),
                        scale = (0.95,1.05),
                        shear = 2.86
                    )
            ],p=0.8)
            ])
        else:
            self.augment_trans=None
    def __getitem__(self, index):
        name = self.names[index]
        img_path = os.path.join(self.image_dir,name)
        mask_path = os.path.join(self.mask_dir,name)
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.transform is not None:
            img,mask = self.transform(img,mask)
        if self.augment:
            img = img
            mask[mask>=125] = 1
            elastic = ut.ExtElasticTransform()
            img,mask = elastic(img,mask)
        else:
            mask[mask>=125] = 1
        return img,mask
    def __len__(self):
        return len(self.names)