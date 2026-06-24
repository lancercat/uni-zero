from typing import Union

import cv2
import torch
from torch import nn, dtype
from torch.nn.modules.module import T

from neko_sdk.cfgtool.argsparse import neko_get_arg

# do NOT concat (so they preserve their size so the model can decide which part it wants zoom in and which part it will zoom out)
# welcome back, TPTNet
class neko_list_rgb_mvn_dev(nn.Module):
    def device(this):
        return this.mean.data.device;
    def get_type(this):
        return this.mean.data.dtype;
    def __init__(this, param):
        super().__init__();
        mean = neko_get_arg("mean", param, [127.5]);
        this.mean_val=mean;
        var = neko_get_arg("var", param, [128]);
        mean_var_img = neko_get_arg("2dstat", param, False);
        if (mean_var_img):
            this.mean = torch.nn.Parameter(torch.tensor(mean).float().squeeze(0), False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().squeeze(0), False);
        else:
            this.mean = torch.nn.Parameter(torch.tensor(mean).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0), False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0), False);

    def forward(this, imagelist):
        imgl = [(torch.tensor(i,dtype=this.get_type()).permute(2, 0, 1).unsqueeze(0).to(
                this.mean.data.device) - this.mean )/ this.var for i in imagelist];
        return imgl;

class neko_list_padding_rgb_mvn_dev(neko_list_rgb_mvn_dev):
    PARAM_padval="padval";
    def __init__(this, param):
        super().__init__(param);
        this.padval=neko_get_arg(this.PARAM_padval,param,3);

    def forward(this, imagelist):
        imgl = [(torch.tensor(cv2.copyMakeBorder(i, this.padval, this.padval, this.padval, this.padval, cv2.BORDER_CONSTANT,
            value=this.mean_val),dtype=this.get_type()).permute(2, 0, 1).unsqueeze(0).to(
                this.mean.data.device) - this.mean )/ this.var for i in imagelist];
        return imgl;


class neko_concat_dev(nn.Module):
    def device(this):
        return this.mean.data.device;
    def get_type(this):
        return this.mean.data.dtype;
    def __init__(this,param):
        super().__init__();
        mean=neko_get_arg("mean",param,[127.5]);
        var=neko_get_arg("var",param,[128]);
        mean_var_img=neko_get_arg("2dstat",param,False);
        if(mean_var_img):
            this.mean=torch.nn.Parameter(torch.tensor(mean).float().squeeze(0),False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().squeeze(0), False);
        else:
            this.mean=torch.nn.Parameter(torch.tensor(mean).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0),False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0), False);

    def forward(this,imagelist):
        if(len(imagelist[0].shape)==3):
            imgt=torch.stack([torch.tensor(i,dtype=this.get_type()) for i in imagelist]).permute(0,3,1,2).contiguous().to(this.mean.data.device)-this.mean;
            if(imgt.shape[1]==1):
                imgt=imgt.repeat([1,3,1,1]); # prototypes can be tricky
        elif(len(imagelist[0].shape)==4):
            imgt = torch.cat(imagelist,dim=0).contiguous().to(
                this.mean.data.device,dtype=this.get_type()) - this.mean;
        else:
            imgt=torch.stack([torch.tensor(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)) for i in imagelist]).permute(0,3,1,2).contiguous().to(this.mean.data.device)-this.mean;
            # well we should add a regularization agent later....

        return imgt/this.var;
    


