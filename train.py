import argparse
import os
import numpy as np
import torch
from unet import UNet
from unet import UNetNested
from utile.deeplearning import train_net
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Trainer(object):
    def __init__(self, args):
        self.args = args





def main():

    parser = argparse.ArgumentParser(description="Pytorch Unet Training")
    parser.add_argument('--backbone', type=str, default='unet',
                        choices=['unet','unetnested','unet_siis'],
                        help="backbone name (default: unet)")
    parser.add_argument('--dataset', type=str, default='D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/',
                        help="dataset dir ")
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop images size')
    parser.add_argument('--loss-type',type=str, default='ce',
                        choices=['ce','focal'],
                        help='loss func type (default: ce)')

    #training hyperparameters
    parser.add_argument('--epochs',type=int, default=40, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for train ')   
    parser.add_argument('--learn-rate', type=float, default=None, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')       
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='w-decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='decay coefficient of learning rate')
    parser.add_argument('--step-size', type=int, default=5, 
                        help='learning rate decay interval')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed ')
        
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt/model/',
                        help='set the checkpoint dir')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separeated list of integers only')

    print(args)



    



    checkpoint_dir = os.path.join("./ckpt/", 'model/') # 模型保存路径
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    # 参数设置
    param = {}
    param['data_dir'] = 'D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/'
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
    train_net(param)

if __name__ == "__main__":
    main()
