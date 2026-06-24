import torch
from torch import nn

from neko_sdk.cfgtool.argsparse import neko_get_arg




# How do you want your attention sampler? static, dynamic, or a spicy mixture?
class attention_conv_sigmoid_mk1(torch.nn.Module):
    PARAM_number_channels="number_channels";
    PARAM_n_parts="n_parts";
    PARAM_maxT="maxT";
    def __init__(this,params):
        super().__init__();
        this.maxT=neko_get_arg(this.PARAM_maxT,params);
        this.n_parts=neko_get_arg(this.PARAM_n_parts,params);
        this.conv=torch.nn.Sequential(
            nn.Conv2d(neko_get_arg(this.PARAM_number_channels,params),
                            this.maxT*this.n_parts,1),
            nn.Sigmoid(),
        );
        this.maxT=params[this.PARAM_maxT]
        pass;
    def forward(this,feats,gfeats=None):
        att=this.conv(feats);
        return att.reshape([att.shape[0],this.maxT,this.n_parts,att.shape[2],att.shape[3]]);


