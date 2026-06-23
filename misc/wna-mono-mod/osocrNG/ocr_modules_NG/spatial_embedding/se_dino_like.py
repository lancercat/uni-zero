import torch
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se,neko_add_embint_se_HD_nocond

class temporal_se_dino_like_mk1(torch.nn.Module):
    PARAM_scales="scales";
    PARAM_num_se_channels="num_se_channels";

    def __init__(this, params):
        super().__init__();
        scales=params[this.PARAM_scales];
        num_se_channels=params[this.PARAM_num_se_channels];
        this.semods=[];
        if(type(num_se_channels) is int):
            for cid in range(len(scales)):
                this.semods.append(neko_add_embint_se(scales[cid][1],scales[cid][2],num_se_channels));
                this.add_module("se_"+str(cid),this.semods[-1]);
        else:
            fatal("error");
    def forward(this,feats):
        return [s(f) for s,f in zip(this.semods,feats)];


class temporal_se_dino_like_HD_no_input(torch.nn.Module):
    PARAM_scales="scales";
    PARAM_num_se_channels="num_se_channels";

    def __init__(this, params):
        super().__init__();
        scales=params[this.PARAM_scales];
        num_se_channels=params[this.PARAM_num_se_channels];
        this.semods=[];
        if(type(num_se_channels) is int):
            for cid in range(len(scales)):
                this.semods.append(neko_add_embint_se_HD_nocond(scales[cid][1]*2,scales[cid][2]*2,num_se_channels));
                this.add_module("se_"+str(cid),this.semods[-1]);
        else:
            fatal("error");
    def forward(this):
        return [s() for s in this.semods];


class temporal_se_dino_like_no_input(torch.nn.Module):
    PARAM_scales="scales";
    PARAM_num_se_channels="num_se_channels";

    def __init__(this, params):
        super().__init__();
        scales=params[this.PARAM_scales];
        num_se_channels=params[this.PARAM_num_se_channels];
        this.semods=[];
        if(type(num_se_channels) is int):
            for cid in range(len(scales)):
                this.semods.append(neko_add_embint_se_HD_nocond(scales[cid][1],scales[cid][2],num_se_channels));
                this.add_module("se_"+str(cid),this.semods[-1]);
        else:
            fatal("error");
    def forward(this):
        return [s() for s in this.semods];

