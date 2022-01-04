import os
import numpy as np
import pandas as pd
import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from PIL import Image

from src.models import *

# load data utils =====================================================================================        
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
        return tensor_image, self.meta[idx], self.labels[idx]

def load_imgs(folder, batch_size=64, val_size=0):
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

# train_val utils =====================================================================================
def train_val(model, optimizer, loss_func, epochs, train_dataloader, val_dataloader, device, verbose=True):
    # main iterations
    print('Training start...')
    best_rmse = np.inf
    for e in tqdm.tqdm(range(epochs)):
        # train
        model.train()
        train_pred = list()
        train_true = list()
        for train_images, train_meta, train_labels in train_dataloader:
            train_images = train_images.to(device).float()
            train_labels = train_labels.to(device).float()
            train_meta = train_meta.to(device).float()

            # forward
            out = model(train_images, train_meta)
            loss = torch.sqrt(loss_func(out, train_labels))

            # backprop
            # print('backprop...')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_true += train_labels.cpu().detach().numpy().tolist()
            train_pred += out.cpu().detach().numpy().tolist()

        train_rmse = calc_rmse(np.array(train_pred), np.array(train_true))
        # verbose
        if verbose:
            print('Train rmse: {}'.format(train_rmse))
        
        # validatin
        if val_dataloader is not None:
            model.eval()
            val_pred = list()
            val_true = list()
            for val_images, val_meta, val_labels in val_dataloader:
                val_images = val_images.to(device).float()
                val_meta = val_meta.to(device).float()
                val_true += val_labels.cpu().detach().numpy().tolist()

                # forward
                val_pred += model.predict(val_images, val_meta).cpu().detach().numpy().tolist()
            rmse = calc_rmse(np.array(val_pred), np.array(val_true))

            # update global vars
            if rmse < best_rmse:
                best_rmse = rmse
            
            # verbose
            if verbose:
                print('Epoch {}, Val rmse: {}, Best rmse: {}'.format(e, rmse, best_rmse))

# evaluation metric ====================================================================
def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
