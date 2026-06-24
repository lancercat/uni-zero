import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se


# derives from previous SE mk3
# NG will have spatial encoding by default, and also multi-headed by default.
# if you just want one head, just set nparts to 1.


class spatial_attention_NG_mk2(nn.Module):
    PARAM_num_se_channels="num_se_channels";
    PARAM_ifc="ifc";
    PARAM_nparts="nparts";
    def set_se_engine(this,params):
        se_channel = neko_get_arg(this.PARAM_num_se_channels, params, 32);
        this.se_engine=neko_add_embint_se(16,16,se_channel);
        pass;
    def set_core(this,params):
        ifc=neko_get_arg(this.PARAM_ifc,params,32);
        se_channel = neko_get_arg(this.PARAM_num_se_channels, params, 32);

        nparts=neko_get_arg(this.PARAM_nparts,params,1);
        this.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc+se_channel, ifc+se_channel, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc+se_channel),
            torch.nn.Conv2d(ifc+se_channel, nparts, (1, 1)),
            torch.nn.Sigmoid(),
        );


    def __init__(this,params):
        super(spatial_attention_NG_mk2, this).__init__();
        this.set_se_engine(params);
        this.set_core(params);

    def forward(this, input):
        x = input[0];
        d=input[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return this.core(this.se_engine(x));
