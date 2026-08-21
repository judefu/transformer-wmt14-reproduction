from torch import nn

class FeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ffn):
        super().__init__()
        self.dense1=nn.Linear(d_model,d_ffn)
        self.dense2=nn.Linear(d_ffn,d_model)
        self.relu=nn.ReLU()

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))