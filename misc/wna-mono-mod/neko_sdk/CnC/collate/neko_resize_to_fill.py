import numpy as np
import torch
from torch import nn
from typing import Tuple, List, Dict,Optional
from torch.nn import functional as trnf
from neko_sdk.cfgtool.argsparse import neko_get_arg

import cv2
# this is not the best friend of static and locality friendly code.
# DONT you DARE try this on an AMD GPU! You have been warned.

# heck why don't we just fill this darn thing now that we have the darn anchors.
# if it bites it should route away. Simple.

def neko_resize_and_padfill_core(tensors: List[torch.Tensor],
                    valid_size:List[int],margins:Optional[List[Tuple[int]]],
                          padval:float=0.,mode:str="bilinear"):
    rt=[trnf.interpolate(t, valid_size,mode=mode) for t in tensors];
    ms=np.array(margins).reshape(-1);
    if(max(ms)==0):
        return rt;
    ms=tuple(ms);
    rt=[trnf.pad(t,ms,value=padval) for t in rt];
    return rt;

# Just tensor. (DDLC reference)
def neko_resize_and_padfill_fn(tensors: List[torch.Tensor], bmasks: Optional[List[torch.Tensor]],
                    valid_size:List[int],margins:Optional[List[int]],
                          padval:float=0.,mode:str="bilinear"):
    if bmasks is not None:
        tarlist=[torch.cat([c,m],dim=1) for c,m in zip(tensors,bmasks)];
    else:
        tarlist=tensors;
    tarlist=neko_resize_and_padfill_core(tarlist,valid_size,margins,padval,mode);
    if(bmasks is not None):
        tensors=tarlist[:,:tensors[0].shape[1]];
        bmasks=tarlist[:,tensors[0].shape[1]:];
        return tensors,bmasks;
    else:
        tensors=tarlist;
        return tensors;
class neko_resize_and_padfill(nn.Module):
    PARAM_target_size="target_size";
    PARAM_margins="margins";
    PARAM_padval="padval";
    PARAM_interpolate_mode="interpolate_mode"
    def __init__(this,param:Dict):
        super().__init__();
        this.target_size = neko_get_arg(this.PARAM_target_size, param);
        this.margins=neko_get_arg(this.PARAM_margins,param,[(4,4),(4,4)]);
        this.padval=neko_get_arg(this.PARAM_padval,param,0);
        this.int_mode=neko_get_arg(this.PARAM_interpolate_mode,param,"bilinear");
        this.valid_size=[t-m[0]-m[1] for t,m in zip(this.target_size,this.margins)];

    def forward(this,tensors,bmasks=None):
        return neko_resize_and_padfill_fn(tensors,bmasks,this.valid_size,this.margins,this.padval,this.int_mode);

# just do one thing a time, let another module/agent to handle padding and resizing


class neko_resize_and_fill(nn.Module):
    PARAM_target_size="target_size";
    PARAM_interpolate_mode="interpolate_mode"
    def __init__(this,param:Dict):
        super().__init__();
        this.target_size = neko_get_arg(this.PARAM_target_size, param);
        this.int_mode=neko_get_arg(this.PARAM_interpolate_mode,param,"bilinear");
    def forward(this,tensors,bmasks=None):
        if bmasks is not None:
            tarlist = [torch.cat([c, m], dim=1) for c, m in zip(tensors, bmasks)];
        else:
            tarlist = tensors;
        tarlist = [trnf.interpolate(t, this.target_size, mode=this.int_mode) for t in tarlist];
        if (bmasks is not None):
            tensors = tarlist[:, :tensors[0].shape[1]];
            bmasks = tarlist[:, tensors[0].shape[1]:];
            return tensors, bmasks;
        else:
            tensors = tarlist;
            return tensors;

# shuffle wxh block gray information to channel. A trick to do "SR" while only burdens the first layer.
class neko_resize_and_fill_gx(nn.Module):
    PARAM_target_size="target_size";
    PARAM_interpolate_mode="interpolate_mode";
    PARAM_factor_w="factor_w";
    PARAM_factor_h="factor_h";
    def __init__(this,param:Dict):
        super().__init__();
        this.target_size = neko_get_arg(this.PARAM_target_size, param);\
        this.fh=neko_get_arg(this.PARAM_factor_h,param);
        this.fw=neko_get_arg(this.PARAM_factor_w,param)
        this.gray_target_size=(this.fh*this.target_size[0],
                               this.fw*this.target_size[1]);
        this.int_mode=neko_get_arg(this.PARAM_interpolate_mode,param,"bilinear");
    def grayshuffle(this,t):
        igt= trnf.interpolate(
            t.mean(dim=1, keepdim=True), this.gray_target_size, mode=this.int_mode);
        return igt.reshape(1, 1, this.target_size[0], this.fh, this.target_size[1], this.fw).permute(0, 3, 5, 1, 2, 4).reshape(1,
                                                                                                                    this.fh * this.fw,
                                                                                                                    this.target_size[
                                                                                                                        0],
                                                                                                                    this.target_size[
                                                                                                                        1])
    def forward(this,tensors,bmasks=None):
        if bmasks is not None:
            tarlist = [torch.cat([ m,c], dim=1) for c, m in zip(tensors, bmasks)];
        else:
            tarlist = tensors;
        tarlist = [torch.cat([this.grayshuffle(t),trnf.interpolate(t, this.target_size, mode=this.int_mode)],1) for t in tarlist];


        if (bmasks is not None):
            tensors = tarlist[:, 1];
            bmasks = tarlist[1:];
            return tensors, bmasks;
        else:
            tensors = tarlist;
            return tensors;

if __name__ == '__main__':
    e=neko_resize_and_padfill;
    a=cv2.imread("/home/lasercat/Pictures/WIN_20230825_02_47_00_Pro.jpg")[:403,:12];

    ei=e({e.PARAM_target_size:(32+4+4,128+4+4)});
    r=ei([ torch.tensor([a]).permute([0,3,1,2]),torch.tensor([a]).permute([0,3,1,2])]);
    cv2.imshow("meow",r.numpy());
    cv2.waitKey(0);



