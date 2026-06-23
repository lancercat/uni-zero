import torch
from neko_sdk.cfgtool.argsparse import neko_get_arg

# add a temperature to
class neko_logit_scaler(torch.nn.Module):
    #    n=nB or n=nB*nT   nxnC       kxnC
    PARAM_init_temp="init_temp";
    def set_para(this):
        this.scale=torch.nn.Parameter(torch.zeros(1).float()+this.init_temp);
    def __init__(this,param):
        super(neko_logit_scaler, this).__init__();
        this.init_temp=neko_get_arg(this.PARAM_init_temp,param,1)
        this.set_para();
    def forward(this,logits):
        return logits*this.scale;
