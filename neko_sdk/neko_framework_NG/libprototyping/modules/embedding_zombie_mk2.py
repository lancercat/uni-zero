import os.path
from typing import Any, Iterator
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf, Parameter
from neko_sdk.log import warn,info,fatal


# a bank of embedding-- dead.
class neko_embedding_zombie_mk2(nn.Module):
    PARAM_libsize="libsize";
    PARAM_num_ch="numch";
    PARAM_parts="num_parts";


    KEY_CUR_UTF_TDICT="utf_tdict";
    KEY_CUR_UTF_LST = "utf_list";
    KEY_UUID="uuid";
    KEY_NUMPK="num_pks";

    DFT_capacity=16384;

    def init_param(this):
        return torch.rand([this.num_cls, this.num_parts, this.num_ch]) * 2 - 1;
    def __init__(this,param):
        super().__init__();
        this.tdict={}; # label of each
        this.uuid={};
        this.utf_list=[];
        this.num_pks=0;
        this.num_cls=neko_get_arg(this.PARAM_libsize,param,this.DFT_capacity);
        this.num_parts=neko_get_arg(this.PARAM_parts,param,1);
        this.num_ch=neko_get_arg(this.PARAM_num_ch,param,512);
        this.core=nn.Parameter(this.init_param());
    def register_one(this,utf,pkid=None):
        if(utf in this.tdict):
            return this.tdict[utf];
        if(pkid is None):
            pkid=this.num_pks;
            this.num_pks+=1;
            this.tdict[pkid]=utf;
        this.tdict[utf]=pkid;
        this.uuid[utf]=len(this.utf_list);
        this.utf_list.append(utf);
        return pkid;
    def load_state_dict(
        this, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        this.utf_list = state_dict[this.KEY_CUR_UTF_LST];
        this.tdict=state_dict[this.KEY_CUR_UTF_TDICT];
        this.uuid=state_dict[this.KEY_UUID];
        this.num_pks=state_dict[this.KEY_NUMPK];

        keys_to_exclude = [this.KEY_CUR_UTF_LST, this.KEY_CUR_UTF_TDICT,this.KEY_UUID,this.KEY_NUMPK];

        new_dict = dict(filter(lambda item: item[0] not in keys_to_exclude, state_dict.items()))
        return super().load_state_dict(new_dict, strict, assign)

    def state_dict(this, *args, destination=None, prefix="", keep_vars=False):
        d= super().state_dict(*args,destination=destination,prefix=prefix,keep_vars=keep_vars);
        d[this.KEY_CUR_UTF_LST]=this.utf_list;
        d[this.KEY_CUR_UTF_TDICT]=this.tdict;
        d[this.KEY_NUMPK]=this.num_pks;
        d[this.KEY_UUID]=this.uuid;

        return d;

    def register_pk(this,utf,pk):
        if (utf in this.tdict):
            return;
        pkid=this.register_one(pk)
        this.register_one(utf,pkid);

    def register_utfs_with_pk(this,utfs, pks):
        for u,pk in zip(utfs,pks):
            this.register_pk(u,pk);
    def register_utfs_as_pk(this,utfs):
        this.register_utfs_with_pk(utfs,utfs);

    def forward(this,utf_list=None):
        if(utf_list is None):
            utf_list= this.utf_list;
            protos=this.core[:len(this.utf_list)];
            tdict=this.tdict;
        else:
            this.register_utfs_as_pk(utf_list);
            return this.forward(None);
        return protos,utf_list,tdict; # if we have less log
class neko_embedding_zombie_normed_mk2(neko_embedding_zombie_mk2):
    def forward(this,utf_list=None):
        proto,utf_list,tdict=super().forward(utf_list);
        return trnf.normalize(proto, dim = -1, p = 2),utf_list,tdict; # if we have less logit


