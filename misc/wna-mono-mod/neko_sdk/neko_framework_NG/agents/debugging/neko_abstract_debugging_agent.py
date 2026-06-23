import os.path

import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace

class neko_abstract_debugging_agent(neko_module_wrapping_agent):
    PARAM_dstpath="dstpath";
    PARAM_saveprfx="prfx";
    INPUT_uid="uid";
    DFT_saveprfx=None;
    DFT_postfx=".pt";
    def uid2fn(this,uid):
        return "-".join([str(uid[k]) for k in uid.keys()]);
    def get_fns(this,workspace):
        items=workspace.get(this.uid);
        fns= [os.path.join(this.dstpath, this.prfx+this.uid2fn(i)+this.DFT_postfx) for i in items] ;
        return fns;
    def get_items(this,workspace):
        fns=this.get_fns(workspace);
        rds=[neko_workspace() for _ in fns];
        for k in this.keys:
            for i in range(len(fns)):
                rds[i].add(k,[workspace.get(k)[i]]);
        return fns,rds;

    def set_etc(this,param):
        this.dstpath=neko_get_arg(this.PARAM_dstpath,param);
        this.prfx=neko_get_arg(this.PARAM_saveprfx,param,this.DFT_saveprfx);
        os.makedirs(this.dstpath,exist_ok=True);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.uid=this.register_input(this.INPUT_uid,iocvt_dict);
        this.keys=None;