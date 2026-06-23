import torch
import torch.nn as nn
class neko_tanh_thresholder_static(nn.Module):
    def __init__(this,params):
        super().__init__();
        this.threshold=nn.Parameter(torch.zeros([1],dtype=torch.float32));

    def forward(this):
        return nn.functional.tanh(this.threshold);