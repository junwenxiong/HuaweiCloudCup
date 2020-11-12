from utile.dataloaders import RSCDataset
from torch.utils.data import DataLoader

img_dir = 'D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/train/images/'
label_dir = 'D:/CodingFiles/Huawei_Competition/Huawei/huawei_data/train/labels/'
train_dataset = RSCDataset(img_dir, label_dir, flag='val')
print(train_dataset.__len__())
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for i, sample in enumerate(train_dataloader):
    data, label = sample['image'], sample['label']
    print(i)