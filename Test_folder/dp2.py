import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

import logging
from glob import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
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

        img = self.preprocess(img)
        mask = np.array(mask)
        return {
            'trace': img.astype(np.float32),
            'label': mask.astype(np.long)
        }


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def train_net(param, model, train_data, valid_data, plot=True):
    # 初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_dir = param['checkpoint_dir']

    train_size = train_data.__len__()
    valid_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['trace'].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_loss = 1e50
    best_mode = copy.deepcopy(model)
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
        train_loss_per_epoch = 0
        for batch_idx, batch_samples in enumerate(tbar):
            data, target = batch_samples['trace'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            pred = model(data)
            # print(pred.shape, target.shape)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            tbar.set_description('Train loss: %.5f' % (loss / (batch_idx + 1)))
            train_loss_per_epoch += loss.item()
        # 验证阶段
        model.eval()
        valid_loss_per_epoch = 0
        tbar = tqdm(valid_loader)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tbar):
                data, target = batch_samples['trace'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                loss = criterion(pred, target)
                valid_loss_per_epoch += loss.item()
                tbar.set_description('Test loss : %.5f' % (loss / (batch_idx + 1)))
        train_loss_per_epoch = train_loss_per_epoch / train_size
        valid_loss_per_epoch = valid_loss_per_epoch / valid_size
        train_loss_total_epochs.append(train_loss_per_epoch)
        valid_loss_total_epochs.append(valid_loss_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        # 保存最优模型
        if valid_loss_per_epoch < best_loss:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = valid_loss_per_epoch
            best_mode = copy.deepcopy(model)
        scheduler.step()
        # 显示loss
        if epoch % disp_inter == 0:
            print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}'.format(epoch, train_loss_per_epoch,
                                                                                  valid_loss_per_epoch))
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='训练集loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='验证集loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title(f'训练曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'学习率变化曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    return best_mode, model


def pred(model, data):
    target_l = 1024
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.transpose(2, 0, 1)
    if data.max() > 1: data = data / 255
    c, x, y = data.shape
    label = np.zeros((x, y))
    x_num = (x // target_l + 1) if x % target_l else x // target_l
    y_num = (y // target_l + 1) if y % target_l else y // target_l
    for i in tqdm(range(x_num)):
        for j in range(y_num):
            x_s, x_e = i * target_l, (i + 1) * target_l
            y_s, y_e = j * target_l, (j + 1) * target_l
            img = data[:, x_s:x_e, y_s:y_e]
            img = img[np.newaxis, :, :, :].astype(np.float32)
            img = torch.from_numpy(img)
            img = Variable(img.to(device))
            out_l = model(img)
            out_l = out_l.cpu().data.numpy()
            out_l = np.argmax(out_l, axis=1)[0]
            label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)
    print(label.shape)
    return label


def cal_metrics(pred_label, gt):
    def _generate_matrix(gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = num_class * gt_image[mask].astype('int') + pre_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class ** 2)
        confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    def _Class_IOU(confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        return MIoU

    confusion_matrix = _generate_matrix(gt.astype(np.int8), pred_label.astype(np.int8))
    miou = _Class_IOU(confusion_matrix)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return miou, acc