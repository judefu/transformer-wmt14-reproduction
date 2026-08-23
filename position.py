import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,maxlen=1000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros(1,maxlen,num_hiddens)
        X=torch.arange(maxlen,dtype=torch.float32).reshape(
            -1,1)/torch.pow(10000,torch.arange(
                0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)        

    def forward(self,X):
        X=X*(X.shape[-1]**0.5)+self.P[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)