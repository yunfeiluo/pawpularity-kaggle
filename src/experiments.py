from src.utils import *
from src.models import *

if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    # hyper-parameter config
    lr = 1e-4
    epochs = 30
    batch_size = 256
    hidden_size = 2048

    latent_out = 524 # 2060

    print('Construct models...')
    model = FineTuneModel(latent_out, hidden_size, 1, device).to(device)
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    #     momentum=0.9, 
    #     nesterov=True
    )
    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()

    print('Construct dataloaders...')
    train_dataloader, val_dataloader = load_imgs('train', batch_size=batch_size, val_size=0)

    train_val(
        model, 
        optimizer, 
        loss_func,
        epochs, 
        train_dataloader, 
        val_dataloader,
        device,
        verbose=True
    )

    # save model
    torch.save(model.cpu().state_dict(), 'trained_model.model')
