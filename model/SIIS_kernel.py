import torch
from torch import nn
from torch.nn import functional as F
import math


class SegRNNCell(nn.Module):
    def __init__(self, dim, ks, bias=True, nonlinearity='relu'):
        super(SegRNNCell, self).__init__()
        self.dim = dim
        self.ks = ks
        self.bias = bias
        self.alpha = nn.Parameter(
            torch.ones((1, 1, 1, 1, 1), dtype=torch.float32))

        self.w_ih = self._make_layers(ks)
        self.w_hh = self._make_layers(ks)
        self.w_hz = self._make_layers((1, 1, 1))

    def _make_layers(self, k_size=(1, 1, 1), stride=1):
        pad = ((k_size[0] - 1) // 2, (k_size[1] - 1) // 2, (k_size[2] - 1) // 2)
        return nn.Sequential(
            nn.Conv3d(
                in_channels=self.dim, out_channels=self.dim,
                kernel_size=k_size, stride=stride, padding=pad, bias=False
            ),
            nn.BatchNorm3d(self.dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        (x, hx) = input
        Wx = self.w_ih(x)
        Wh = self.w_hh(hx)
        ht = Wx + self.alpha * Wh
        hz = self.w_hz(ht)
        return hz


class BuildPass_4(nn.Module):
    def __init__(self, d, width, kw, dim, Num):
        super(BuildPass_4, self).__init__()
        self.d = d
        self.dim = dim
        self.kw = kw
        self.width = width
        self.num = Num // width
        assert (self.num * width) == Num

        # 设置两个convs通道（从下到上 或从左到右）
        self.pass_1 = nn.Sequential()
        self.pass_2 = nn.Sequential()

        if self.d == 1:
            self.pass_1.add_module(('SIIS_D'),
                                   SegRNNCell(dim, (width, 1, kw), False))
            self.pass_2.add_module(('SIIS_U'),
                                   SegRNNCell(dim, (width, 1, kw), False))
        else:
            self.pass_1.add_module(('SIIS_D'),
                                   SegRNNCell(dim, (width, kw, 1), False))
            self.pass_2.add_module(('SIIS_U'),
                                   SegRNNCell(dim, (width, kw, 1), False))

        std = math.sqrt(2 / (self.width * self.kw * self.dim * self.dim * 5))

        # 下去在查一下资料
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, std)

    def forward(self, x):
        # [NCHW] -> n*[N, C, width, W] or n*[N,C,H,width]
        fm = torch.split(x, self.width, self.d + 1)
        if len(fm) != self.num:
            raise ValueError(
                'The number of feature map cuts is inconsistent with the number of filters'
            )

        FM = []

        # 1.Down/Left first
        for i in range(self.num):
            # Feature: [N,C, width, W] -> [N, C, width, 1, W]
            # Filter kernel : [width, kw] -> [width, 1, kw] or [width, kw] -> [width, kw,1]
            FM.append(fm[i].unsqueeze(2).transpose(2, self.d + 2))
        fm = []
        hx = FM[0].detach()
        for i in range(0, self.num):
            hx = self.pass_1((FM[i], hx))
            FM[i] = hx

        # 2. UP/Right second
        hx = FM[-1].detach()
        for i in range(self.num - 2, -1, -1):
            hx = self.pass_2((FM[i], hx))
            FM[i] = hx

        # n * [N,C,width, 1, W] - > [NCH1W]->H[NC11W]
        FM = torch.split(torch.cat(FM, 2), 1, 2)
        # 5D -> 4d
        FM = [s.transpose(2, self.d + 2).squeeze(2) for s in FM]
        # H*[NC1W] -> [NCHW]
        x = torch.cat(FM, self.d + 1)
        return x


class SIIS_Conv3dRNN(nn.Module):
    def __init__(self, input_shape, width=1, kw=9, dim=128):
        super(SIIS_Conv3dRNN, self).__init__()
        self.name = 'SIIS_Conv3dRNN'
        self.input_shape = input_shape
        self.new_size = None
        [H, W] = input_shape
        while H % width != 0:
            H += 1
            self.new_size = [H, W]
        while W % width != 0:
            W += 1
            self.new_size = [H, W]
        self.DU = BuildPass_4(1, width, kw, dim, H)
        self.LR = BuildPass_4(2, width, kw, dim, W)

    def forward(self, x):
        if self.new_size is not None:
            x = F.interpolate(x, size=self.new_size, mode='nearest')
            x = self.DU(x)
            x = self.LR(x)
            x = F.interpolate(x, size=self.input_shape, mode='nearest')
        else:
            x = self.DU(x)
            x = self.LR(x)
        return x


class SIIS_Conv1d(nn.Module):
    def __init__(self, input_shape, width=1, kw=9, dim=128):
        super(SIIS_Conv1d, self).__init__()
        self.name = 'SIIS_Conv1d'
        self.input_shape = input_shape
        self.dim = dim
        self.kw = kw

        self.D=self._make_layer()
        self.U = self._make_layer()

        self.L = self._make_layer()
        self.R = self._make_layer()


    def _make_layer(self):
        return nn.Sequential(
            nn.Conv1d(self.dim, self.dim, self.kw ,1 ,padding=(self.kw-1)//2),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[:2]

        fms = torch.split(x, 1, dim=3)
        FMs= []
        for fm in fms:
            new_fm = self.D(fm.squeeze(3))
            new_fm = self.U(flip(new_fm, 2))
            FMs.append(flip(new_fm, 2))
        x = torch.stack(FMs, dim=2)

        fms = torch.split(x,1,dim=2)
        FMs = []
        for fm in fms:
            new_fm =self.L(fm.squeeze(2))
            new_fm = self.R(flip(new_fm, 2))
            FMs.append(flip(new_fm, 2))
        x = torch.stack(FMs, dim=2)

        return x

#flip tensor along the specific dim
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim

    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]

    return x.view(xsize)


def SIIS(input_shape, width=1, kw=9, dim=128, arch=1):
    if arch == 4:
        return SIIS_Conv3dRNN(input_shape, width, kw, dim)
    elif arch == 7:
        return SIIS_Conv1d(input_shape, width, kw, dim)


if __name__ == "__main__":
    print(" ####Test Case ### ")
    from torch.autograd import Variable
    x = Variable(torch.randn(2,3,256,256))
    model = SIIS(input_shape=[256, 256], width=3, kw=9, dim=3, arch=7)
    y = model(x)
    print('Output shape:', y.shape)
    print(model)