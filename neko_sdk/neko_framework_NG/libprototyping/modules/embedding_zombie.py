import os.path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf
from neko_sdk.log import warn,info,fatal
import copy
from collections import OrderedDict
# it is living, but its dead--- hence zombie
# the reason of the zombie state can be found in utf_embedding_iho.py
class neko_embedding_zombie(nn.Module):
    PARAM_max_cnt="max_cnt";
    PARAM_num_ch="numch";

    KEY_CUR_UTF_TDICT="utf_tdict";
    KEY_CUR_UTF_LST = "utf_list";

    def init_param(this):
        return torch.rand([this.curbufsz, 1, this.num_ch]) * 2 - 1;
    # this thing has a different loading behaviour due to the growing pattern.
    def load_state_dict(
        this, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        this.utfs = state_dict[this.KEY_CUR_UTF_LST];
        this.tdict=state_dict[this.KEY_CUR_UTF_TDICT];
        keys_to_exclude = [this.KEY_CUR_UTF_LST, this.KEY_CUR_UTF_TDICT]
        new_dict = dict(filter(lambda item: item[0] not in keys_to_exclude, state_dict.items()))
        return super().load_state_dict(new_dict, strict, assign)

    def state_dict(this, *args, destination=None, prefix="", keep_vars=False):
        d= super().state_dict(*args,destination=destination,prefix=prefix,keep_vars=keep_vars);
        d[this.KEY_CUR_UTF_LST]=this.utfs;
        d[this.KEY_CUR_UTF_TDICT]=this.tdict;

        return d;


    def __init__(this,param):
        super().__init__();
        this.tdict={};
        this.utfs=[];
        this.curbufsz=neko_get_arg(this.PARAM_max_cnt,param,16384); # bcs the optimizer want the parameter unchanced --- bite the bullet and preallocate.
        this.num_ch=neko_get_arg(this.PARAM_num_ch,param,512);
        this.core=nn.Parameter(this.init_param());
    def update_centers(this,utfs):
        with torch.no_grad():
            for u in utfs:
                if(u not in this.tdict):
                    this.tdict[u]=len(this.utfs);
                    this.tdict[len(this.utfs)]=u;
                    this.utfs.append(u);
            lutf=len(this.utfs);
            if(lutf>this.curbufsz):
                delta=int(np.ceil((lutf-this.curbufsz)/this.chunksz))*this.chunksz;
                oldsz=this.curbufsz;
                this.curbufsz+=delta;
                data=this.init_param();
                data[:oldsz]=this.core;
        if(lutf>this.curbufsz):
            this.core=nn.Parameter(data);
    def forward(this,utfs):
        if(utfs and this.training):
            this.update_centers(utfs);
        return this.core[:len(this.utfs)], copy.copy(this.utfs);


if __name__ == '__main__':
    def loss(p):
        return p.sum();

    m=neko_embedding_zombie({
        neko_embedding_zombie.PARAM_max_cnt:16384,
    });

    a1=m(["cat"]);
    a2=m(["dog"]);
    a6=m(["cat","ape"]);
    a3=m(["meow","hiss","purr"]);
    a4=m(["hiss"]);
    a5=m(["dog"]);
    a=m.state_dict();
    torch.save(m.state_dict(),"a.pt");
    mm=neko_embedding_zombie({
        neko_embedding_zombie.PARAM_max_cnt:16384,
    });
    torch.save(m.state_dict(), "a.pt");
    mm.load_state_dict(torch.load("a.pt"));
    pass;



