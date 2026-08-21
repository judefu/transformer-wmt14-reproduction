import torch
import math
from torch import nn
from torch.nn import functional as F

class DotProductAttention(nn.Module):
    def __init__(self,valid_lens=None):
        super().__init__()

