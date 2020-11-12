import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
#import moxing as mox
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utile.evaluator import Evaluator
from Test_folder.TestData import TestDataset

from unet import UNet

from apex import amp

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def train_net(param,  plot=True):
    #初始化参数
    epochs         = param['epochs']
    batch_size     = param['batch_size']
    lr             = param['lr']
    gamma          = param['gamma']
    step_size      = param['step_size']
    momentum       = param['momentum']
    weight_decay   = param['weight_decay']
    disp_inter     = param['disp_inter']
    save_inter     = param['save_inter']
    checkpoint_dir = param['checkpoint_dir']
    data_dir       = param['data_dir']

    train_imgs_dir = os.path.join(data_dir, "train/images/")
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    val_labels_dir = os.path.join(data_dir, "val/labels/")

    # using Dataset obeject written by myself
    # train_data = RSCDataset(train_imgs_dir, train_labels_dir, flag='train')
    # valid_data = RSCDataset(val_imgs_dir, val_labels_dir, flag='val')

    # using Dataset obeject provided by sponsor
    train_data =TestDataset(train_imgs_dir, train_labels_dir)
    valid_data = TestDataset(val_imgs_dir, val_labels_dir)



    train_size = train_data.__len__()
    valid_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

    model = UNet(3, 2).to(device)

    evaluator = Evaluator(2)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # 1. modified the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

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
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            target = target.long()
            optimizer.zero_grad()
            pred = model(data)
            # print(pred.shape, target.shape)
            loss = criterion(pred, target)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            optimizer.step()
            tbar.set_description('Train loss: %.5f' % (loss / (batch_idx + 1)))
            train_loss_per_epoch += loss.item()
        # caculate the miou
        pred_out = pred.data.cpu().numpy()
        target_out = target.cpu().numpy()
        # why
        pred_out = np.argmax(pred_out, axis=1)
        evaluator.add_batch(pred_out, target_out)

        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()

        train_loss_per_epoch = train_loss_per_epoch / train_size
        train_loss_total_epochs.append(train_loss_per_epoch)
        print("train  validation")
        print('epoch:{}, loss:{}, train_mIou:{}, train_acc:{}'.format(epoch+1, train_loss_per_epoch, mIoU, Acc_class))

        # 验证阶段

        model.eval()
        evaluator.reset()
        valid_loss_per_epoch = 0
        tbar = tqdm(valid_loader)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tbar):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                target = target.long()
                loss = criterion(pred, target)
                valid_loss_per_epoch += loss.item()
                tbar.set_description('Test loss : %.5f' % (loss / (batch_idx + 1)))
                pred = pred.data.cpu().numpy()
                target = target.cpu().numpy()
                # retrun the indices of the maximum values along the axis
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)

        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()

        valid_loss_per_epoch = valid_loss_per_epoch / valid_size
        valid_loss_total_epochs.append(valid_loss_per_epoch)
        print('test validation')
        print('epoch:{}, loss:{},  val_mIoU:{}, val_acc:{}'.format(epoch+1, valid_loss_per_epoch,  mIoU, Acc_class))

        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        # 保存最优模型
        if valid_loss_per_epoch < best_loss: # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = valid_loss_per_epoch
            best_mode = copy.deepcopy(model)
        scheduler.step()
        # # 显示loss
        # if epoch % disp_inter == 0: 
        #     print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}'.format(epoch+1, train_loss_per_epoch, valid_loss_per_epoch))
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
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'学习率变化曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()



def pred(model, data):
    target_l = 1024
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.transpose(2, 0, 1)
    if data.max() > 1: data = data / 255
    c, x, y = data.shape
    label = np.zeros((x, y))
    x_num = (x//target_l + 1) if x%target_l else x//target_l
    y_num = (y//target_l + 1) if y%target_l else y//target_l
    for i in tqdm(range(x_num)):
        for j in range(y_num):
            x_s, x_e = i*target_l, (i+1)*target_l
            y_s, y_e = j*target_l, (j+1)*target_l
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



