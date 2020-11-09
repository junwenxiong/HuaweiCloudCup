import torch
import random
import numpy as np
import cv2 as cv

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """

    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['image'] = img
        sample['label'] = mask

        return sample


class ToTensor(object):
    def __call__(self, sample):
        #swap color axis because
        #numpy image: HxWxC
        #torch image: C x H x W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        sample['image'] = img
        sample['label'] = mask

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.25:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.25:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img, 'label': mask}


class RandomRotate(object):
    def __init__(self, ):
        self.degree = [0, 90, 180, 270]

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        index = random.randint(0, 3)
        if index != 0:
            rotate_degree = self.degree[index]
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.BILINEAR)

        return {'imaage': img, 'label': mask}


class RandomGammaTransform(object):
    def __call__(self, sample):
        img = sample['image']
        img_np = np.array(img, dtype=np.uint8)
        alpha = np.random.uniform(-np.e, np.e)
        gamma = np.exp(alpha)
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        img_np = cv.LUT(img_np, gamma_table)
        img = Image.fromarray(img_np, mode='CMYK')
        sample['image'] = img
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=random.randint(2, 5)))

        return {'image': img, 'label': mask}


class RandomNoise(object):
    def __call__(self, sample):
        img = sample['image']
        w, h = img.size
        img_np = np.array(img, dtype=np.uint8)
        for i in range(5000):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            img_np[x, y] = 255
        img = Image.fromarray(img_np, mode='CMYK')
        sample['image'] = img
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        short_size = random.randint(int(self.base_size * 0.5),
                                    int(self.base_size * 2.0))

        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask,
                                   border=(0, 0, padw, padh),
                                   fill=self.fill)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)

        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, 'label': mask}
