import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg

class one_normed_embedding(nn.Module):
    PARAM_channel="channel";
    def __init__(this,params):
        super().__init__();
        this.emb=torch.nn.Parameter(torch.rand([1,neko_get_arg(this.PARAM_channel,params)]));


    def forward(this):
        return trnf.normalize(this.emb,dim=1,p=2);
class neko_class_embeddingNG(nn.Module):
    PARAM_channel="channel";
    PARAM_n_parts="n_parts"
    PARAM_transcriptions="transcriptions"
    def __init__(this,params):
        super().__init__();
        this.trans=neko_get_arg(this.PARAM_transcriptions,params);

        this.emb=torch.nn.Parameter(torch.rand([len(this.trans),neko_get_arg(this.PARAM_n_parts,params),
                                                neko_get_arg(this.PARAM_channel,params)]));



    def forward(this):
        return trnf.normalize(this.emb,dim=-1,p=2),this.trans;
