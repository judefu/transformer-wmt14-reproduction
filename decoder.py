import torch
from torch import nn
import addnorm 
import attention
import feedforwardnet
import position

class DecoderBlock(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,
                 value_size,dropout,d_ffn):
        super().__init__()
        self.attention1=attention.MultiHeadAttention(num_heads,num_hiddens,
                                                     query_size,key_size,value_size)
        self.addnorm1=addnorm.AddNorm(num_hiddens,dropout)
        self.attention2=attention.MultiHeadAttention(num_heads,num_hiddens,
                                                     query_size,key_size,value_size)
        self.addnorm2=addnorm.AddNorm(num_hiddens,dropout)
        self.ffn=feedforwardnet.FeedForwardNet(num_hiddens,d_ffn)
        self.addnorm3=addnorm.AddNorm(num_hiddens,dropout)

    def forward(self,X,state):
        dec_valid_lens=torch.arange(1,X.shape[1]+1,device=X.device
                                         ).repeat(X.shape[0],1)
        Y=self.attention1(X,X,X,dec_valid_lens)
        Z=self.addnorm1(Y,X)
        enc_outputs,enc_valid_lens=state[0],state[1]
        Y2=self.attention2(Z,enc_outputs,enc_outputs,enc_valid_lens)
        Z2=self.addnorm2(Y2,Z)
        return self.addnorm3(self.ffn(Z2),Z2)
        

class Decoder(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,embedding,
                 value_size,dropout,d_ffn,vocab_size,num_layers):
            super().__init__()
            self.embedding=embedding
            self.pos_encoding=position.PositionalEncoding(num_hiddens,dropout)
            self.blks=nn.Sequential()
            for i in range(num_layers):
                 self.blks.add_module("block"+str(i),
                                      DecoderBlock(num_heads,num_hiddens,
                                                   query_size,key_size,value_size,
                                                   dropout,d_ffn))
            self.dense=nn.Linear(num_hiddens,vocab_size)
            self.dense.weight=self.embedding.weight
            self.num_layers=num_layers

    def init_state(self,enc_outputs,enc_valid_len):
        return [enc_outputs,enc_valid_len]
         
    def forward(self,X,state):
        X=self.embedding(X)
        X=self.pos_encoding(X)
        for blk in self.blks:
             X=blk(X,state)
        return self.dense(X)