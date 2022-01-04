from src.utils import *
from src.models import *

class PreTrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # with open('pretrained_models/resnext101_32x8d.pkl', 'rb') as f:
        with open('pretrained_models/resnet18.pkl', 'rb') as f:
            pretrain_model = pickle.load(f)
            self.pretrain_feat = nn.Sequential(*(list(pretrain_model.children())[:-1]))
            self.pretrain_class = nn.Sequential(*(list(pretrain_model.children())[-1:]))
    
    def forward(self, x):
        feat_out = self.pretrain_feat(x).squeeze()
        class_out = self.pretrain_class(feat_out)
        return feat_out, class_out

class PawpularityPreTrainedDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def load_pretrained_data(folder, device, batch_size=64, val_size=0):
    # fetch labels
    meta_data = pd.read_csv("dataset/{}.csv".format(folder))
    labels = list()
    if folder == 'train':
        labels = meta_data['Pawpularity'].to_numpy()
    else:
        labels = [None for i in range(len(meta_data))]
    labels = np.array(labels)
    
    # load pretrained out
    if folder == 'test':
        samples, classes = preprocess_with_pretrained_model('test', device)
    else:
        with open('resnext101_32x8d_out.pkl', 'rb') as f:
            pretrained_data = pickle.load(f)
            samples = pretrained_data['features']
    
    # split if test_size > 0
    if val_size > 0:
        inds = [i for i in range(len(labels))]
        np.random.shuffle(inds)
        split_ind = int(len(inds) * val_size)
        train_inds = inds[split_ind:]
        val_inds = inds[:split_ind]

    # construct dataset and dataloader    
    if folder == 'test':
        dataset = PawpularityPreTrainedDataset(samples=samples, labels=labels)
        dataloader = DataLoader(dataset, batch_size=len(labels), shuffle=False)
        
        samples, ls = next(iter(dataloader))
        return samples, [i for i in meta_data['Id']]
    else:
        if val_size == 0:
            train_dataset = PawpularityPreTrainedDataset(samples=samples, labels=labels)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # change shuflle here if do not wanna shuffle
            return train_dataloader
        
        train_dataset = PawpularityPreTrainedDataset(samples=samples[train_inds], labels=labels[train_inds])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # change shuflle here if do not wanna shuffle
        
        val_dataset = PawpularityPreTrainedDataset(samples=samples[val_inds], labels=labels[val_inds])
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_inds), shuffle=False)
        
        return train_dataloader, val_dataloader

# preprocess with pretrained model ===================================================
def preprocess_with_pretrained_model(folder, device):
    # construct model
    model = PreTrainedModel().to(device)

    # iterate through data
    if folder == 'test':
        dataiter = iter(load_pretrained_data('test', batch_size=16, val_size=0))
    else:
        dataiter = iter(load_pretrained_data('train', batch_size=16, val_size=0))

    features = list()
    labels = list()
    classes = list()

    while True:
        try:
            images, meta, label = dataiter.next()
            images = images.to(device)
            labels += label.cpu().detach().numpy().tolist()

            # forward
            feat_out, class_out = model(images)
            features += np.concatenate((feat_out.cpu().detach().numpy(), meta), axis=1).tolist()
            classes += class_out.argmax(dim=1).squeeze().cpu().detach().numpy().tolist()
        # break
        except:
            break
    
    classes = np.array(classes)
    features = np.array(features)

    # save the result
    with open('pretrained_name.pkl', 'wb') as f:
        pickle.dump({'features': features, 'labels': labels, 'classes': classes}, f)
    
    return features, labels, classes

if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
