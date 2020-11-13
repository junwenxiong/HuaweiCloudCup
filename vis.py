import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from args_config import make_args
from utile.dataloaders import make_data_loader
from unet.unet_model import UNet
from unet.unetNested import UNetNested
from unet.unet_SIIS import UNet_SIIS


class Visualization(object):
    def __init__(self, args):
        self.args = args

        kwags = {'num_workers': self.args.workers, 'pin_memory': True}
        _, self.test_loader, _, _ = make_data_loader(self.args, **kwags)

        self.model = None

        if self.args.backbone == 'unet':
            self.model = UNet(n_channels=3, n_classes=2)
            print('using UNet')
        if self.args.backbone == 'unetnested':
            self.model = UNetNested(in_channels=3, n_classes=2)
            print('using UNetNested')
        if self.args.backbone == 'unetsiis':
            self.model = UNet_SIIS(n_channels=3, n_classes=2)
            print('using UNet_SIIS')

        if self.args.cuda:
            self.model = self.model.cuda()

        if not os.path.isfile(self.args.checkpoint_dir):
            raise RuntimeError("=> no checkpoint found as '{}'".format(
                self.args.checkpoint_dir))
        checkpoint = torch.load(self.args.checkpoint_dir)

        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.args.checkpoint_dir))

    def visualization(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image = sample['image']

            if self.args.cuda:
                image = image.cuda()
                with torch.no_grad():
                    output = self.model(image)
                tbar.set_description('Vis image:')
                pred = output.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)[0]

                pred_img = Image.fromarray(pred, mode='L')
                filename = 'label_{}.png'.format(i)
                vis_dir = 'D:\\CodingFiles\\Huawei_Competition\\HuaweiCloudCup\\Test_folder\\labels'
                pred_img.save(os.path.join(vis_dir, filename))


def main():
    args = make_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.checkpoint_dir = 'D:\\CodingFiles\\Huawei_Competition\\Test_Files\\11_13_unet\\checkpoint_epoch8_mIoU_0.5227712844956314.pth'
    # args.checkpoint_dir = 'D:/CodingFiles/Huawei_Competition/Test_Files/11_12_unet/model_best.pth'
    args.dataset = 'D:\\CodingFiles\\Huawei_Competition\\Test_Files\\huawei_data\\'
    args.batch_size = 1
    args.backbone = 'unetnested'


    visual = Visualization(args)
    visual.visualization()


if __name__ == "__main__":
    main()