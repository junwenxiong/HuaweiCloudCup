import os
import copy
import torch
import torch.nn as nn
import logging
import numpy as np
from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utile.dataloaders import custom_transform as tr


class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, flag='train'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.flag = flag
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        sample = {'image':img, 'label':mask}
        if self.flag == 'train':
            return self.transform_train(sample)
        if self.flag == 'val':
            return self.transform_val(sample)

        # img = self.preprocess(img)
        # mask = np.array(mask)
        # return {
        #     'trace': img.astype(np.float32),
        #     'label': mask.astype(np.long)
        # }


    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            # tr.RandomGammaTransform(),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.544650, 0.352033, 0.384602,), std=(0.249456, 0.241652, 0.228824,)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.544650, 0.352033, 0.384602,), std=(0.249456, 0.241652, 0.228824,)),
            tr.ToTensor()])
        return composed_transforms(sample)



if __name__ == "__main__":
    img_dir = 'D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/train/images/'
    label_dir = 'D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/train/labels/'
    dataset = RSCDataset(img_dir, label_dir)
    print(dataset.__len__())