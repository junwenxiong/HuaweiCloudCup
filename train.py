from model import UNet
from model import UNetNested
import utile.deeplearning as dl
import os
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # data_dir = "./data/"
    # train_imgs_dir = os.path.join(data_dir, "train/images/")
    # val_imgs_dir = os.path.join(data_dir, "val/images/")
    # train_labels_dir = os.path.join(data_dir, "train/labels/")
    # val_labels_dir = os.path.join(data_dir, "val/labels/")
    # train_data = dl.RSCDataset(train_imgs_dir, train_labels_dir)
    # valid_data = dl.RSCDataset(val_imgs_dir, val_labels_dir)
    checkpoint_dir = os.path.join("./ckpt/", 'model/') # 模型保存路径
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    # 参数设置
    param = {}
    param['data_dir'] = './data/'
    param['epochs'] = 41       # 训练轮数
    param['batch_size'] = 1   # 批大小
    param['lr'] = 1e-2         # 学习率
    param['gamma'] = 0.9       # 学习率衰减系数
    param['step_size'] = 5     # 学习率衰减间隔
    param['momentum'] = 0.9    #动量
    param['weight_decay'] = 0. #权重衰减
    param['checkpoint_dir'] = checkpoint_dir
    param['disp_inter'] = 1 # 显示间隔
    param['save_inter'] = 1 # 保存间隔
    # 训练
    print(param)
    best_model, model = dl.train_net(param)

if __name__ == "__main__":
    train()
