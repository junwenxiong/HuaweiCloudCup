from model.unet_model import UNet
from model.unetNested import UNetNested
import torch
from torch.autograd import Variable

x = Variable(torch.rand(2,3,64,64))
# model = UNet(3,2)
model = UNetNested(3,2)
y = model(x)
print("output shape{}".format(y.shape))
print(model)