from src.utils import *
from src.models import *

if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    # config
    batch_size = 64
    epochs = 30

    print('Construct models...')
    model = FineTuneModel().to(device)
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    #     momentum=0.9, 
    #     nesterov=True
    )
    loss_func = nn.MSELoss()

    print('Construct dataloaders...')
    train_dataloader, val_dataloader = load_data('train', batch_size=batch_size, val_size=0.2)

    train_val(
        model, 
        optimizer, 
        loss_func,
        epochs, 
        train_dataloader, 
        val_dataloader,
        verbose=True
    )

    # save model
    torch.save(model.cpu().state_dict(), 'trained_model.model')
