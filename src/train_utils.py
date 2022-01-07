import numpy as np
import tqdm

# train_val utils =====================================================================================
def train_val(
    model, 
    optimizer, 
    loss_func, 
    epochs, 
    train_dataloader, 
    val_dataloader, 
    device, 
    verbose=True
):

    # main iterations
    print('Training start...')
    best_eval = np.inf
    train_loss = list()
    for e in tqdm.tqdm(range(epochs)):
        # train
        model.train()
        for data_pack in train_dataloader:
            # move to device
            for key in data_pack:
                data_pack[key] = data_pack[key].to(device).float()

            # forward
            loss = model(data_pack, loss_func=loss_func)

            # backprop
            # print('backprop...')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updata var
            train_loss.append(loss.cpu().detach().item())

        # verbose
        if verbose:
            print('Train Epoch {}, last loss {}, best loss {}.'.format(e, train_loss[-1], np.array(train_loss).min()))
        
        # validation
        if val_dataloader is not None:
            model.eval()
            val_pred = list()
            val_true = list()
            for data_pack in val_dataloader:
                # move to device
                for key in data_pack:
                    data_pack[key] = data_pack[key].to(device).float()

                # foward
                pred, true = model.predict(data_pack)

                # uypdate var
                val_true += true.cpu().detach().numpy().tolist()
                val_pred += pred.cpu().detach().numpy().tolist()

            eval_ = calc_rmse(np.array(val_pred), np.array(val_true))

            # update global vars
            if eval_ < best_eval:
                best_eval = eval_
            
            # verbose
            if verbose:
                print('Epoch {}, Val eval: {}, Best eval: {}'.format(e, eval_, best_eval))

# evaluation metric ====================================================================
def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
