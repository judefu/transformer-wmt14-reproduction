from torch import nn
import decoder
import encoder

class Transformer(nn.Module):
    def __init__(self,num_heads,num_hiddens,query_size,key_size,
                 value_size,dropout,d_ffn,num_layers,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.encoder=encoder.Encoder(num_heads,num_hiddens,query_size,key_size,
                                     self.embedding,value_size,dropout,d_ffn,num_layers)
        self.decoder=decoder.Decoder(num_heads,num_hiddens,query_size,key_size,
                                     self.embedding,value_size,dropout,d_ffn,
                                     vocab_size,num_layers)

    def forward(self,X,Y,valid_lens):
        enc_outputs=self.encoder(X,valid_lens)
        state=self.decoder.init_state(enc_outputs,valid_lens)
        return self.decoder(Y,state)