import pickle
import torch
from torch import nn
from torchvision import models

class Regressor(nn.Module):
    def __init__(
        self, 
        in_size=524, # 2060
        hidden_size=2048, 
        out_size=1
    ):
        super().__init__()
                
        self.fc_liner = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.Softplus(),
        )
    
    def forward(self, data_pack, loss_func=None):
        samples = data_pack['samples']
        out = self.fc_liner(samples).squeeze()

        # compute loss
        if loss_func is not None:
            labels = data_pack['labels']
            return loss_func(out, labels)
        return out
    
    def predict(self, data_pack):
        # unpack
        labels = data_pack['labels']
        
        return self.forward(data_pack), labels

class ResNet18(nn.Module):
    def __init__(self, finetune=False):
        super().__init__()

        # load pretrained model (download pretrained model here if needed)
        with open('pretrained_models/resnet18.pkl', 'rb') as f:
            pretrain_model = pickle.load(f)
            self.pretrain_feat = nn.Sequential(*(list(pretrain_model.children())[:-1]))
        
        # make the model fine-tuned
        for param in self.pretrain_feat.parameters():
            param.requires_grad = finetune
    
    def forward(self, x):
        return self.pretrain_feat(x)

class IntegratedModel(nn.Module):
    def __init__(self, feature_extractor=None, regressor=None):
        super().__init__()

        self.feature_extractor = ResNet18(finetune=True) if feature_extractor is None else feature_extractor
        
        # construct the final output layer(s)
        self.regressor = Regressor() if regressor is None else regressor

    def forward(self, data_pack, loss_func=None):
        # unpack
        imgs = data_pack['images']
        meta = data_pack['meta']

        # forward
        feat_out = self.feature_extractor(imgs).squeeze()

        # do something with meta data
        N, D = feat_out.shape
        N, M = meta.shape
        out = torch.zeros((N, D+M)).to(self.device)
        out[:, :D] = feat_out
        out[:, D: ] = meta

        # final out
        regressor_pack = {'samples': out}
        out = self.regressor(regressor_pack).squeeze()

        # compute loss
        if loss_func is not None:
            labels = data_pack['labels']
            return loss_func(out, labels)
        return out
    
    def predict(self, data_pack):
        # unpack
        labels = data_pack['labels']

        return self.forward(data_pack), labels

