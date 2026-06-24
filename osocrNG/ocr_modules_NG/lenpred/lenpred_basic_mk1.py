
import torch
from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg

class lenpred_basic_mk1(torch.nn.Module):
    PARAM_input_channels="input_channels";
    PARAM_maxT="maxT";
    PARAM_allow_empty="allow_empty";

    def __init__(this,params):
        super().__init__();
        lchs = neko_get_arg(this.PARAM_input_channels, params);
        all_emp=neko_get_arg(this.PARAM_allow_empty,params,True);
        this.lenpred=nn.Linear(lchs,neko_get_arg(this.PARAM_maxT,params)+all_emp);
    def forward(this,feat):
        return this.lenpred(feat);

