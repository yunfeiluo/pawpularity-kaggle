from src.train_utils import *
from src.models import *
from src.data_loader import *

def get_pretrained_out(feature_extractor, train_dataloader, val_dataloader, batch_size, device, reg=True):
    print('Pretrained model forwarding on train set...')
    train_samples = list()
    train_labels = list()
    for data_pack in train_dataloader:
        # move to device
        for key in data_pack:
            data_pack[key] = data_pack[key].to(device).float()

        # unpack
        imgs = data_pack['images']
        labels = data_pack['labels']

        print('forwarding...')
        feat_out = feature_extractor(imgs).cpu().detach().numpy()

        meta = data_pack['meta']
        feat_out = np.concatenate((feat_out, meta.cpu().detach().numpy()), axis=1)

        train_samples += feat_out.tolist()
        train_labels += labels.cpu().detach().numpy().tolist()
    # construct data loader
    train_pre_dataset = SimpleDataset(np.array(train_samples), np.array(train_labels))
    train_pre_loader = DataLoader(train_pre_dataset, batch_size=batch_size, shuffle=True)
    
    print('Pretrained model forwarding on val set...')
    if val_dataloader is not None:
        val_samples = list()
        val_labels = list()
        for data_pack in val_dataloader:
            # move to device
            for key in data_pack:
                data_pack[key] = data_pack[key].to(device).float()

            # unpack
            imgs = data_pack['images']
            labels = data_pack['labels'] if reg else data_pack['class_labels']

            print('forwarding...')
            feat_out = feature_extractor(imgs).cpu().detach().numpy()

            meta = data_pack['meta']
            feat_out = np.concatenate((feat_out, meta.cpu().detach().numpy()), axis=1)

            val_samples += feat_out.tolist()
            val_labels += labels.cpu().detach().numpy().tolist()
        # construct data loader
        val_pre_dataset = SimpleDataset(np.array(val_samples), np.array(val_labels))
        val_pre_loader = DataLoader(val_pre_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_pre_loader = None

    return train_pre_loader, val_pre_loader

def train_model(regressor, train_loader, val_loader, loss_func, device, epochs=30, lr=1e-4):
    optimizer = torch.optim.Adam(
        regressor.parameters(),
        lr=1e-4,
    #     momentum=0.9, 
    #     nesterov=True
    )

    train_val(
        regressor, 
        optimizer, 
        loss_func,
        epochs, 
        train_loader, 
        val_loader,
        device,
        verbose=True
    )


if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    # config
    batch_size = 256
    print('batch size', batch_size)

    print('Construct dataloaders...')
    train_dataloader, val_dataloader = load_data('train', batch_size=batch_size, val_size=0)

    print('Construct feature extractor...')
    feature_extractor = ResNet18().to(device)
    print('Num params:', sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))
    
    # get pretrained out
    train_pre_loader, val_pre_loader = get_pretrained_out(feature_extractor, train_dataloader, val_dataloader, batch_size, device)

    # print('Train Regressor...')
    # regressor = Regressor().to(device)
    # loss_func = nn.MSELoss()
    # print('Num params:', sum(p.numel() for p in regressor.parameters() if p.requires_grad))
    # # train_model(regressor, train_pre_loader, val_pre_loader, loss_func, device, epochs=30, lr=1e-4)

    print('Train Regressor...')
    regressor = MultitaskOut().to(device)
    loss_func = {
        'reg_loss': nn.MSELoss(),
        'class_loss': nn.CrossEntropyLoss(),
    }
    print('Num params:', sum(p.numel() for p in regressor.parameters() if p.requires_grad))
    train_model(regressor, train_pre_loader, val_pre_loader, loss_func, device, epochs=50, lr=1e-4)

    print('Construct Integrated Model...')
    model = IntegratedModel(device, regressor=regressor).to(device)
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # train_model(model, train_dataloader, val_dataloader, loss_func, device, epochs=1, lr=1e-5)

    # save model
    torch.save(model.cpu().state_dict(), 'trained_model.model')

