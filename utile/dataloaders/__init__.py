import os
from utile.dataloaders.Data import RSCDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def make_data_loader(args, **kwargs):
    data_dir = args.dataset
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    val_labels_dir = os.path.join(data_dir, "val/labels/")


    train_data = RSCDataset(train_imgs_dir, train_labels_dir, flag='train')
    valid_data = RSCDataset(val_imgs_dir, val_labels_dir, flag='val')
    train_kwargs = kwargs
    val_kwargs = kwargs

    # if use mixed precision for training
    if args.mixed_precision:
        train_sampler = DistributedSampler(train_data)
        val_sampler = DistributedSampler(valid_data)
        train_kwargs['sampler'] = train_sampler
        val_kwargs['sampler'] = val_sampler
        train_kwargs['shuffle'] = train_sampler is None

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, **train_kwargs)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size,  **val_kwargs)

    train_len = train_data.__len__()
    val_len = valid_data.__len__()

    return train_loader, valid_loader, train_len, val_len 
