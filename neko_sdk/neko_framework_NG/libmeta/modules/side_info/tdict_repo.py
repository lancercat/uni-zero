
import os.path

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.file_names import FN

from torch import nn
import torch
import copy

def load_tdict(path):
    return torch.load(path,weights_only=False); # we don't need magic now. Hope we don't

class neko_tdict_repo(nn.Module):
    PARAM_root="root";
    PARAM_name="file_name"; # leave it if you are not performing magic... 
    CONST_default_name=FN.TDICT;

    def __init__(this, param):
        super().__init__();
        this.path = os.path.join(neko_get_arg(this.PARAM_root, param),
                                 neko_get_arg(this.PARAM_name, param, this.CONST_default_name));
        this.tdict= load_tdict(this.path);

    def forward(this):
        return copy.copy(this.tdict);
