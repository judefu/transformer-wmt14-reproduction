import torch
from torch import nn
from torch.nn import functional as F

def masked_softmax(X,valid_lens=None):
    if valid_lens is None:
        return F.softmax(X,dim=-1)    
    shape=X.shape
    if valid_lens.dim()==1:
        valid_lens=torch.repeat_interleave(valid_lens,repeats=shape[-2]).to(X.device)
    else:
        valid_lens=valid_lens.reshape(-1).to(X.device)
    X=X.reshape(-1,shape[-1])
    mask=torch.arange(shape[-1],device=X.device)[None,:]>=valid_lens[:,None] 
    X=X.masked_fill(mask,-1e6)
    return F.softmax(X.reshape(shape),dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,query,key,value,valid_lens=None):
        d=query.shape[-1]
        scores=(query@key.transpose(-2,-1))/(d**0.5)
        scores=masked_softmax(scores,valid_lens)
        return scores@value

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,value_size):
        super().__init__()
        self.W_q=nn.Linear(query_size,num_hiddens)
        self.W_k=nn.Linear(key_size,num_hiddens)
        self.W_v=nn.Linear(value_size,num_hiddens)
        self.W_o=nn.Linear(num_hiddens,num_hiddens)
        self.num_heads=num_heads
        self.attention=DotProductAttention()

    def forward(self,query,key,value,valid_lens=None):
        transpose_qkv=lambda x:x.reshape(x.shape[0],x.shape[1],self.num_heads,-1
                                         ).permute(0,2,1,3)
        query=transpose_qkv(self.W_q(query))
        key=transpose_qkv(self.W_k(key))
        value=transpose_qkv(self.W_v(value))

        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(valid_lens,
                                               repeats=self.num_heads,dim=0)

        output=self.attention(query,key,value,valid_lens).permute(0,2,1,3)
        output=output.reshape(output.shape[0],output.shape[1],-1)
        return self.W_o(output)