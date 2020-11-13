import argparse

def make_args():
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
    parser.add_argument('--mixed_precision', default='no_use',
                        type=str,
                        choices=['use', 'no_use'],
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
    parser.add_argument('--optim', default='sgd',
                        type=str,
                        choices=['adam', 'sgd'],
                        help='optimizer type ')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
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
    return args
