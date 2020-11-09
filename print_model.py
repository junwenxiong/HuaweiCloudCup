from model.unet_model import UNet
from model.unetNested import UNetNested
import torch
from torch.autograd import Variable
from model.util import count_param
from model.unet_SIIS import UNet_SIIS
def UNet_print():
    x = Variable(torch.rand(2,3,64,64))
    # model = UNet(3,2)
    model = UNet(3,2)
    y = model(x)
    param = count_param(model)
    print("output shape{}".format(y.shape))
    print('UNet total parameters: %.2fM (%d)' % (param /1e6, param))
    # print(model)

def UNet_Nested_print():
    x = Variable(torch.rand(2,3,64,64))
    model = UNetNested(3,2)
    param = count_param(model)
    y = model(x)
    # print(model)
    print("output shape{}".format(y.shape))
    print('UNetNested total parameters: %.2fM (%d)' % (param /1e6, param))

def UNet_SIIS_print():
    x = Variable(torch.rand(2,3,64,64))
    model = UNet_SIIS(3,2)
    y = model(x)
    param = count_param(model)
    print('output shape{}'.format(y.shape))
    print('UNet_SIIS total parameters: %.2fM (%d)' % (param /1e6, param))
    print(model)


def print_model_info(model, input):
    output = model(input)
    param = count_param(model)
    print('output shape{}'.format(output.shape))
    print('UNet_SIIS total parameters: %.2fM (%d)' % (param /1e6, param))
    # print(model)



if __name__ == "__main__":
    x = Variable(torch.rand(2,3,64,64))

    unet_model = UNet(3,2)
    unetnested_model = UNetNested(3,2)
    unet_siis = UNet_SIIS(3,2)

    print_model_info(unet_model, x)
    print_model_info(unetnested_model, x)
    print_model_info(unet_siis, x)
    print(unet_siis)