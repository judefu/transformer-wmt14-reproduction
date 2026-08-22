import torch
from torch import nn
import addnorm 
import attention
import feedforwardnet
import position

class EncoderBlock(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,
                 value_size,valid_lens,dropout,d_ffn):
        super().__init__()
        self.attention=attention.MultiHeadAttention(num_heads,num_hiddens,query_size,
                                                    key_size,value_size,valid_lens)
        self.addnorm1=addnorm.AddNorm(num_hiddens,dropout)
        self.ffn=feedforwardnet.FeedForwardNet(num_hiddens,d_ffn)
        self.addnorm2=addnorm.AddNorm(num_hiddens,dropout)

    def forward(self,X):
        Y=self.addnorm1(self.attention()) ## X还没填
        Z=self.addnorm2(self.ffn(Y))
        return Z

class Encoder(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,
                 value_size,valid_lens,dropout,d_ffn,vocab_size,num_layers):
            super().__init__()
            self.embedding=nn.Embedding(vocab_size,num_hiddens)
            self.pos_encoding=position.PositionalEncoding(num_hiddens,dropout)
            self.blks=nn.Sequential()
            for i in range(num_layers):
                 self.blks.add_module("block"+str(i),
                                      EncoderBlock(num_heads,num_hiddens,query_size,
                                                   key_size,value_size,valid_lens,
                                                   dropout,d_ffn))

    def forward(self,X):
         X=self.embedding(X)
         X=self.pos_encoding(X)
         self.attention_weights=[None]*self.num_layers
         for i,blk in enumerate(self.blks):
               X=blk(X)