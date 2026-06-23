import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg


# mk2 nomore manages sampling and detaching. all managed out side.
class spatial_attention_NG_mk2(nn.Module):
    PARAM_ifc="ifc";
    PARAM_nparts="nparts";

    def set_core(this,params):
        ifc=neko_get_arg(this.PARAM_ifc,params,32);

        nparts=neko_get_arg(this.PARAM_nparts,params,1);
        this.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc, ifc, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc, nparts, (1, 1)),
            torch.nn.Sigmoid(),
        );


    def __init__(this,params):
        super(spatial_attention_NG_mk2, this).__init__();
        this.set_core(params);

    def forward(this, input):
        return this.core(input);
