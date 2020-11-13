import  os
import shutil
import torch
from collections import OrderedDict
import glob
import time

class Saver(object):
    def __init__(self, args):
        self.args = args

        now = time.strftime("%m-%d-%H_%M_%S", time.localtime())
        self.directory = os.path.join(args.checkpoint_dir, args.backbone, now)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.experiment_dir = os.path.join(args.checkpoint_dir, 'experiments_{}_{}'.format('mixed' if args.mixed_precision else 'normal', args.backbone), now)
        if not  os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, mIoU, loss):
        epoch = state['epoch']
        filename = os.path.join(self.directory, 'checkpoint_epoch{}_mIoU_{}.pth'.format(epoch, mIoU))
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a') as f:
                f.write(str("epoch:{}, best_pred:{}, val_loss:{}".format(epoch, best_pred, loss)))
                f.write('\r\n')  # next line

            shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['learn_rate'] = self.args.learn_rate
        p['weight_decay'] = self.args.weight_decay
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
     