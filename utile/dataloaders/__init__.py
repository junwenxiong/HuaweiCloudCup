import os
from utile.dataloaders.Data import RSCDataset
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    data_dir = args.dataset
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    val_labels_dir = os.path.join(data_dir, "val/labels/")
    train_data = RSCDataset(train_imgs_dir, train_labels_dir, flag='train')
    valid_data = RSCDataset(val_imgs_dir, val_labels_dir, flag='val')
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    train_len = train_data.__len__()
    val_len = valid_data.__len__()

    return train_loader, valid_loader, train_len, val_len 
