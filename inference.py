from collections import OrderedDict
import os
import torch
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from unet import UNet
from unet import UNetNested
from unet import UNet_SIIS
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000


def infer(model_path, model_name, model):

    print('model_name:{}'.format(model_name))
    use_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU for inference')
        use_cuda = True
        checkpoint = torch.load(model_path, map_location=device)

        model = model.to(device)
    else:
        print('Using CPU for inference')
        checkpoint = torch.load(model_path, map_location='cpu')

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
        # print(k)
    model.load_state_dict(new_state_dict)

    # print(checkpoint['state_dict'])


class Inference(object):
    def __init__(self, model_name, model_path, data_path):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.data_dict = {}
        self.data_dict = make_data_dict(self.data_path)
        

        if self.model_name == 'unet':
            self.model = UNet(3, 2)
        elif self.model == 'unetnested':
            self.model = UNetNested(3, 2)
        elif self.model_name == 'unet_siis':
            self.model = UNet_SIIS(3, 2)

        self.use_cuda = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('Using GPU fro inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = self.model.to(self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = np.array(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        img = data['input_image']
        data = img
        target_l = 1024
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.transpose(2, 0, 1)
        if data.max() > 1: data = data / 255
        c, x, y = data.shape
        label = np.zeros((x, y))
        x_num = (x // target_l + 1) if x % target_l else x // target_l
        y_num = (y // target_l + 1) if y % target_l else y // target_l
        for i in range(x_num):
            for j in range(y_num):
                x_s, x_e = i * target_l, (i + 1) * target_l
                y_s, y_e = j * target_l, (j + 1) * target_l
                img = data[:, x_s:x_e, y_s:y_e]
                img = img[np.newaxis, :, :, :].astype(np.float32)
                img = torch.from_numpy(img)
                img = Variable(img.to(self.device))
                out_l = self.model(img)
                out_l = out_l.cpu().data.numpy()
                out_1 = np.argmax(out_1, axis=1)[0]
                label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)

        _label = label.astype(np.int8).tolist()
        _len, __len = len(_label), len(_label[0])
        o_stack = []
        for _ in _label:
            out_s = {"s": [], "e": []}
            j = 0
            while j < __len:
                if _[j] == 0:
                    out_s['s'].append(str(j))
                    while j < __len and _[j] == 0:
                        j += 1
                    out_s["e"].append(str(j))
                j += 1
            o_stack.append(out_s)
        result = {"result": o_stack}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000

        data = self._inference(data)
        infer_end_time = time.time()

        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        data = self._postprocess(data)
        post_time_in_ms = (time.time() - infer_end_time) * 100

        latency_time = str(pre_time_in_ms + infer_in_ms + post_time_in_ms)
        print('latency_time:{}'.format(latency_time))
        return data


def make_data_dict(data_path ):
    data_dict = {}
    temp = {}
    for i, file_list in enumerate(os.listdir(data_path)):
        temp[i] = os.path.join(data_path, file_list)
    data_dict['input_image'] = temp
    return data_dict

if __name__ == "__main__":
    data_path = 'D:\\CodingFiles\\Huawei_Competition\\Huawei\\huawei_data\\val\\one_img'
    unetnested_model_path = 'D:/CodingFiles/Huawei_Competition/Test_Files/UNetNested-epoch5/checkpoint-best.pth'
    unet_model_path = 'D:/CodingFiles/Huawei_Competition/Test_Files/11_12_unet/model_best.pth'
    unet2_model_path = 'D:/CodingFiles/Huawei_Competition/Test_Files/11_12_unet/unet_2433_1_adam.pth'

   

    unet_131 = 'D:\\DownloadFiles\\Thunder\\Huawei_pth\\model_epoch5.pth'
    unet_119 = 'D:\\DownloadFiles\\Thunder\\Huawei_pth\\checkpoint-best.pth'

    unet_nested_model = UNetNested(3, 2)
    unet_model = UNet(3, 2)

    # infer(unet_model_path, 'unet', unet_model)
    # infer(unet2_model_path, 'unet', unet_model)

    # infer(unet_131, 'unet', unet_model)

    # infer(unetnested_model_path, 'unetnested', unet_nested_model)

    # data_dict = make_data_dict(data_path)
    # print(data_dict)

    infer = Inference('unet', unet2_model_path, data_path)
    print(infer.inference(infer.data_dict))