import argparse
import os
import numpy as np
import torch
from args_config import make_args
from tqdm import tqdm
from unet import UNet
from unet import UNetNested
from unet import UNet_SIIS
from utile.dataloaders import make_data_loader
from utile.loss import SegmentationLosses
from utile.evaluator import Evaluator
from utile.saver import Saver
from utile.summaries import TensorboardSummary
from torch.autograd import Variable

from apex import amp


from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        kwags = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.train_size, self.valid_size = make_data_loader(
            args, **kwags)
        
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # import pdb
        # pdb.set_trace()
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


        train_params = [{'params': model.parameters()}]

        #Define Optimizer

        if self.args.optim == 'adam':
            Optimizer = torch.optim.Adam(train_params,
                                         lr=self.args.learn_rate,
                                         weight_decay=self.args.weight_decay,
                                         amsgrad=True)
        elif self.args.optim == 'sgd':
            Optimizer = torch.optim.SGD(train_params, lr=self.args.learn_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        self.model, self.optimizer = model, Optimizer

        #Define Criterion
        weight = None
        self.criterion = SegmentationLosses(
            weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)


        # use Dataparallel 
        if len(self.args.gpu_ids)>1:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=self.args.gpu_ids)
        if self.args.cuda:                                               
            self.model = self.model.cuda()

        #using the apex for speeding up training
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

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

            #using apex for speeding up training
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
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

        # when model is saved, remove the module layer for loading state easily
        self.model_state_dict = self.model.module.state_dict() if len(self.args.gpu_ids) > 1 else self.model.state_dict()

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model_state_dict ,
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

    args = make_args()
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

    print('Total Epochs:', trainer.args.epochs)
    print('-------------------------------')
    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == "__main__":
    main()
