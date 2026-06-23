import torch
from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf

class neko_referenced_unk_confidence_appender(nn.Module):
    PARAM_dim="dim";
    def __init__(this,params):
        super().__init__();
        this.thresh=nn.Parameter(torch.tensor([0],dtype=torch.float32),True);
        this.dim=neko_get_arg(this.PARAM_dim,params,-1);
        pass;
    def forward(this,confidences,references):
        cwunk=torch.cat(
          [confidences,
          trnf.tanh(this.thresh)*torch.norm(references,dim=this.dim,p=2,keepdim=True)],
            dim=this.dim
        );
        return cwunk;