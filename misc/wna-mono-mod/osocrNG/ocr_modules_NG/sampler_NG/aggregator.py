import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se


# derives from previous SE mk3
# NG will have spatial encoding by default, and also multi-headed by default.
# if you just want one head, just set nparts to 1.

class neko_basic_attn_aggr(nn.Module):
    def __init__(this,params):
        super(neko_basic_attn_aggr, this).__init__();

    def forward(this, features,A):
       if(len(A.shape)==4):
           A_=A.unsqueeze(2).unsqueeze(3);
       else:
           A_=A.unsqueeze(3);
        #A_: N,T,P,1,W,H
       return (A_*features.unsqueeze(1).unsqueeze(1)).sum(-1).sum(-1)/A_.sum(-1).sum(-1)

    # N,1,1,CWH