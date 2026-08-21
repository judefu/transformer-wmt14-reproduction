from torch import nn

class AddNorm(nn.Module):
    def __init__(self,normalized_shape,dropout):
        super().__init__()
        self.ln=nn.LayerNorm(normalized_shape)
        self.dropout=nn.Dropout(dropout)

    def forward(self,X,Y):
        return self.ln(self.dropout(X)+Y)