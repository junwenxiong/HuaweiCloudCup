import torch
import torch.nn as nn
from .unet_parts import *
from model.layers import *
from model.util import count_param

class UNetNested(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_ds=True, sync_bn=False):
        super(UNetNested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv 
        self.is_ds = is_ds
        self.batchnorm = nn.BatchNorm2d

        filters = [128, 256, 512, 1024]
        filters = [int(i / self.feature_scale) for i in filters]

        #downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = UnetConv2(self.in_channels, 32, self.batchnorm)
        self.conv10 = UnetConv2(32, 64, self.batchnorm)
        self.conv20 = UnetConv2(64, 128, self.batchnorm)
        self.conv30 = UnetConv2(128, 256, self.batchnorm)
        self.conv40 = UnetConv2(256, 512, self.batchnorm)

        # upsampling
        self.up_concat01 = UnetUp(64, 32, self.is_deconv)
        self.up_concat11 = UnetUp(128, 64, self.is_deconv)
        self.up_concat21 = UnetUp(256, 128, self.is_deconv)
        self.up_concat31 = UnetUp(512, 256, self.is_deconv)

        self.up_concat02 = UnetUp(64, 32, self.is_deconv, 3)
        self.up_concat12 = UnetUp(128, 64, self.is_deconv, 3)
        self.up_concat22 = UnetUp(256, 128, self.is_deconv, 3)

        self.up_concat03 = UnetUp(64, 32, self.is_deconv, 4)
        self.up_concat13 = UnetUp(128, 64, self.is_deconv, 4)

        self.up_concat04 = UnetUp(64, 32 , self.is_deconv, 5)

        self.final_1 = nn.Conv2d(32, 2, 1)
        self.final_2 = nn.Conv2d(32, 2, 1)
        self.final_3 = nn.Conv2d(32, 2, 1)
        self.final_4 = nn.Conv2d(32, 2, 1)

        self.__init_weight()

    def forward(self, inputs):
        
        x_00 = self.conv00(inputs)
        maxpool0 = self.maxpool(x_00)
        x_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool(x_10)
        x_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool(x_20)
        x_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool(x_30)
        x_40 = self.conv40(maxpool3)

        # column:1
        x_01 = self.up_concat01(x_10, x_00)
        x_11 = self.up_concat11(x_20, x_10)
        x_21 = self.up_concat21(x_30, x_20)
        x_31 = self.up_concat31(x_40, x_30)

        #column:2
        x_02 = self.up_concat02(x_11, x_00, x_01)
        x_12 = self.up_concat12(x_21, x_10, x_11)
        x_22 = self.up_concat22(x_31, x_20, x_21)

        # column:3
        x_03 = self.up_concat03(x_12, x_00, x_01, x_02)
        x_13 = self.up_concat13(x_22, x_10, x_11, x_12)

        # column:4
        x_04 = self.up_concat04(x_13, x_00, x_01, x_02, x_03)


        final_1 = self.final_1(x_01)
        final_2 = self.final_2(x_02)
        final_3 = self.final_3(x_03)
        final_4 = self.final_4(x_04)

        final = (final_1+final_2+final_3+final_4) / 4 

        if self.is_ds:
            return final
        else:
            return final_4


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.UpsamplingBilinear2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            

if __name__ == "__main__":
    model = UNetNested()
    param = count_param(model)
    # print(model)
    print('UNetNested total parameters: %.2fM (%d)' % (param /1e6, param))