import os.path

import torch
from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf

# a bank of embedding-- dead.
class neko_abstract_meta_repo(nn.Module):
    PARAM_root="root";
    PARAM_name="name";
    CONST_default_name="nep.nep"; # nonsense name, if you see it that means things went wrong.
    CONST_listname="utf_list";

    def __init__(this, param):
        super().__init__();
        this.path = os.path.join(neko_get_arg(this.PARAM_root, param),
                                 neko_get_arg(this.PARAM_name, param, this.CONST_default_name));
        this.meta= torch.load(this.path);
        this.utf_list=this.meta[this.CONST_listname];
    def mk_proto(this):
        return None;
    def forward(this):
        return this.mk_proto(),this.utf_list;
class neko_utf_dict_meta_repo_mk2(nn.Module):
    PARAM_root="root";
    PARAM_name="name";
    CONST_default_name="nep.nep"; # nonsense name, if you see it that means things went wrong.

    def __init__(this, param):
        super().__init__();
        this.path = os.path.join(neko_get_arg(this.PARAM_root, param),
                                 neko_get_arg(this.PARAM_name, param, this.CONST_default_name));
        this.sideinfo_dict= torch.load(this.path);
        this.utf_list=sorted(list(this.sideinfo_dict.keys()));
    def forward(this):
        return [this.sideinfo_dict[k] for k in this.utf_list],this.utf_list;
