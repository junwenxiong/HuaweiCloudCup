import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from unet import UNet
from unet import UNetNested
from unet import UNet_SIIS
from utile.deeplearning import train_net
from utile.dataloaders import make_data_loader
from utile.loss import SegmentationLosses
from utile.evaluator import Evaluator
from utile.saver import Saver
from utile.summaries import TensorboardSummary
from torch.autograd import Variable
from torch import distributed, optim
from torch.utils.data.distributed import DistributedSampler

# Mixed Precision training
from apex import amp
# synchronous BN
from apex.parallel import convert_syncbn_model
# Disitributed DataParallel
from apex.parallel import DistributedDataParallel as DDP

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer(object):
    def __init__(self, args):
        self.args = args


        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
        torch.cuda.set_device(args.local_rank)


        kwags = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.train_size, self.valid_size = make_data_loader(
            args, **kwags)
        
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        model = None
        # Define network
        if self.args.backbone == 'unet':
            model = UNet(n_channels=3, n_classes=2)
            print('using UNet')
        elif self.args.backbone == 'unetnested':
            model = UNetNested(in_channels=3, n_classes=2)
            print('using UNetNested')
        elif self.args.backbone == 'unet_siis':
            model = UNet_SIIS(n_channels=3, n_classes=2)
            print('using UNet_SIIS')


        # the order must be this
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.model = convert_syncbn_model(model).to(self.device)
        train_params = [{'params': self.model.parameters()}]
        # Define Optimizer
        Optimizer = optim.Adam(train_params,
                               lr=self.args.learn_rate,
                               weight_decay=self.args.weight_decay,
                               amsgrad=True)

        self.model, self.optimizer = amp.initialize(self.model, Optimizer, opt_level='O1')
        self.model = DDP(self.model)


        #Define Criterion
        weight = None
        self.criterion = SegmentationLosses(
            weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

    
        self.evaluator = Evaluator(2)

        self.best_pred = 0.0

        self.checkpoint_dir = args.checkpoint_dir

    def training(self, epoch):
        print('training ')
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            data, target = sample['image'], sample['label']
            if self.args.cuda:
                data, target = Variable(data.cuda()), Variable(target.cuda())
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(),
                                    i +num_img_tr * epoch)


        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        self.evaluator.add_batch(target, pred)

        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('train/train_loss_epoch', train_loss, epoch)

        print('train validation:')
        print('epoch:{}, Loss:{:.3f}, Acc_class:{}, mIoU:{}'.format(
            epoch, train_loss, Acc_class, mIoU))
        print('-------------------------------')

    def validation(self, epoch):
        test_loss = 0.0
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader)

        for i, sample in enumerate(tbar):
            data, target = sample['image'], sample['label']
            if self.args.cuda:
                data, target = Variable(data.cuda()), (target.cuda())
            with torch.no_grad():
                output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss:%.5f' % (test_loss / (i + 1)))
            self.writer.add_scalar('val/total_loss_iter', loss.item(),
                                    i + num_img_val * epoch)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)

        print('test validation:')
        print('epoch:{}, Loss:{:.3f}, Acc_class:{}, mIoU:{}'.format(
            epoch, test_loss, Acc_class, mIoU))
        print('-------------------------------')

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred
        }
        #save checkpont when satisfy the condition only
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint(
                state,
                is_best
            )


def main():

    parser = argparse.ArgumentParser(description="Pytorch Unet Training")
    parser.add_argument('--backbone',
                        type=str,
                        default='unet',
                        choices=['unet', 'unetnested', 'unet_siis'],
                        help="backbone name (default: unet)")
    parser.add_argument('--dataset',
                        type=str,
                        default='./data/',
                        help="dataset dir ")
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        metavar='N',
                        help='dataloader threads')
    parser.add_argument('--base-size',
                        type=int,
                        default=1024,
                        help='base image size')
    parser.add_argument('--crop-size',
                        type=int,
                        default=1024,
                        help='crop images size')
    parser.add_argument('--loss-type',
                        type=str,
                        default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--mixed_precision', default=True,
                        help='whether to use mixed precision')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='ranking within the nodes')


    #training hyperparameters
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=2,
                        metavar='N',
                        help='input batch size for train ')
    parser.add_argument('--learn-rate',
                        type=float,
                        default=1e-2,
                        metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=5e-4,
                        metavar='M',
                        help='w-decay')
    parser.add_argument('--nesterov',
                        action='store_true',
                        default=True,
                        help='whether use nesterov')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='decay coefficient of learning rate')
    parser.add_argument('--step-size',
                        type=int,
                        default=5,
                        help='learning rate decay interval')

    # cuda, seed and logging
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids',
                        type=str,
                        default='0',
                        help='use which gpu to train')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed ')

    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./ckpt/',
                        help='set the checkpoint dir')

    parser.add_argument('--eval-interval',
                        type=int,
                        default=1,
                        help='evaluation interval')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separeated list of integers only'
            )

    print(args)

    trainer = Trainer(args)
    print('Start Training Epoch:', 1)
    print('Total Epochs:', trainer.args.epochs)
    print('-------------------------------')
    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == "__main__":
    main()
