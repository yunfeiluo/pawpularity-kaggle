import os
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image

class SimpleDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_pack = {
            'samples': self.samples[idx],
            'labels': self.labels[idx]
        }

        return data_pack

class PawpularityDataset(Dataset):
    def __init__(self, main_dir, imgs, labels, meta, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = imgs
        self.labels = labels
        self.meta = meta

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        data_pack = {
            'images': tensor_image,
            'meta': self.meta[idx],
            'labels': self.labels[idx]
        }
        return data_pack

def load_data(folder, batch_size=64, val_size=0):
    # fetch filenames
    main_dir = 'dataset/{}'.format(folder)
    img_paths = os.listdir(main_dir)
    meta_data = pd.read_csv("dataset/{}.csv".format(folder))
    if folder == 'test':
        features = meta_data.columns[1:]
    else:
        features = meta_data.columns[1:-1]
    
    # fetch labels
    if folder == 'train':
        labels_dict = dict()
        for i in range(len(meta_data)):
            labels_dict[meta_data['Id'][i]] = meta_data['Pawpularity'][i]

        labels = list()
        for img in img_paths:
            labels.append(labels_dict[img.split('.')[0]])
    else:
        labels = [None for i in range(len(img_paths))]
    
    # fetch meta data
    meta_dict = dict()
    meta = meta_data[features].to_numpy()
    for i in range(len(meta_data)):
        meta_dict[meta_data['Id'][i]] = meta[i]
    meta = list()
    for img in img_paths:
        meta.append(meta_dict[img.split('.')[0]])
    
    img_paths = np.array(img_paths)
    labels = np.array(labels)
    meta = np.array(meta)
    
    # split if test_size > 0
    if val_size > 0:
        inds = [i for i in range(len(img_paths))]
        np.random.shuffle(inds)
        split_ind = int(len(inds) * val_size)
        train_inds = inds[split_ind:]
        val_inds = inds[:split_ind]

    # declare preprocess (add augmentation here)
    train_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # construct dataset and dataloader    
    if folder == 'test':
        dataset = PawpularityDataset(main_dir=main_dir, imgs=img_paths, labels=labels, meta=meta, transform=val_preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return dataloader
    else:
        if val_size == 0:
            train_dataset = PawpularityDataset(main_dir=main_dir, imgs=img_paths, labels=labels, meta=meta, transform=train_preprocess)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # change shuflle here if do not wanna shuffle
            return train_dataloader, None
        
        train_dataset = PawpularityDataset(main_dir=main_dir, imgs=img_paths[train_inds], labels=labels[train_inds], meta=meta[train_inds], transform=train_preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # change shuflle here if do not wanna shuffle
        
        val_dataset = PawpularityDataset(main_dir=main_dir, imgs=img_paths[val_inds], labels=labels[val_inds], meta=meta[val_inds], transform=val_preprocess)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_dataloader, val_dataloader