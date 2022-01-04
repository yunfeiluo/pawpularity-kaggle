import pickle
import torch
from torch import nn
from torchvision import models

class Regressor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
                
        self.fc_liner = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.Softplus(),
        )
    
    def forward(self, x):
        return self.fc_liner(x).squeeze()
    
    def predict(self, x):
        return self.forward(x)

class FineTuneModel(nn.Module):
    def __init__(self, latent_size, hidden_size, out_size, device):
        super().__init__()
        self.device = device

        # load pretrained model (download pretrained model here if needed)
        with open('pretrained_models/resnet18.pkl', 'rb') as f:
            pretrain_model = pickle.load(f)
            self.pretrain_feat = nn.Sequential(*(list(pretrain_model.children())[:-1]))
        
        # make the model fine-tuned
        for param in self.pretrain_feat.parameters():
            param.requires_grad = False
        
        # construct the final output layer(s)
        self.regressor = Regressor(latent_size, hidden_size, out_size)

        # load pretrained mlp if available
        # self.regressor.load_state_dict(torch.load('../input/pawpularity-resnet18-2layer-mlp/resnet18_2layer_mlp.model'))
    
    def forward(self, x, meta):
        feat_out = self.pretrain_feat(x).squeeze()
        # do something with meta data
        N, D = feat_out.shape
        N, M = meta.shape
        out = torch.zeros((N, D+M)).to(self.device)
        out[:, :D] = feat_out
        out[:, D: ] = meta
        return self.regressor(out).squeeze()
    
    def predict(self, x, meta):
        return self.forward(x, meta)

