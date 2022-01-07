from src.train_utils import *
from src.models import *
from src.data_loader import *

def get_pretrained_out(feature_extractor, train_dataloader, val_dataloader, device):
    print('Pretrained model forwarding on train set...')
    train_samples = list()
    train_labels = list()
    for data_pack in train_dataloader:
        imgs = data_pack['images']
        labels = data_pack['labels']

        train_samples += feature_extractor(imgs).cpu().detach().numpy().tolist()
        train_labels += labels.cpu().detach().numpy().tolist()
    train_pre_loader = SimpleDataset(train_samples, train_labels, device)
    
    print('Pretrained model forwarding on val set...')
    if val_dataloader is not None:
        val_samples = list()
        val_labels = list()
        for data_pack in val_dataloader:
            imgs = data_pack['images']
            labels = data_pack['labels']

            val_samples += feature_extractor(imgs).cpu().detach().numpy().tolist()
            val_labels += labels.cpu().detach().numpy().tolist()
        val_pre_loader = SimpleDataset(val_samples, val_labels, device)
    else:
        val_pre_loader = None
    return train_pre_loader, val_pre_loader

def train_model(regressor, train_loader, val_loader, epochs=30, lr=1e-4):
    optimizer = torch.optim.Adam(
        regressor.parameters(),
        lr=1e-4,
    #     momentum=0.9, 
    #     nesterov=True
    )
    loss_func = nn.MSELoss()

    train_val(
        regressor, 
        optimizer, 
        loss_func,
        epochs, 
        train_loader, 
        val_loader,
        verbose=True
    )


if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    # config
    batch_size = 64

    print('Construct dataloaders...')
    train_dataloader, val_dataloader = load_data('train', batch_size=batch_size, val_size=0.2)

    print('Construct feature extractor...')
    feature_extractor = ResNet18().to(device)
    print('Num params:', sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))
    
    # get pretrained out
    train_pre_loader, val_pre_loader = get_pretrained_out(feature_extractor, train_dataloader, val_dataloader, device)

    print('Train Regressor...')
    regressor = Regressor()
    print('Num params:', sum(p.numel() for p in regressor.parameters() if p.requires_grad))
    train_model(regressor, train_pre_loader, val_pre_loader, epochs=10, lr=1e-4)

    print('Construct Integrated Model...')
    model = IntegratedModel(feature_extractor=feature_extractor, regressor=regressor)
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_model(model, train_dataloader, val_dataloader, epochs=10, lr=1e-4)

    # save model
    torch.save(model.cpu().state_dict(), 'trained_model.model')

