import torch
import torch.nn.functional as trnf
from neko_sdk.OSR.classification.linear_classifier import neko_openset_linear_classifier
from neko_sdk.neko_score_merging import scatter_cvt
from torch import nn

# takes plabel
class neko_cosconf_sim_NG(nn.Module):
    def set_para(this,param):
        return;

    def __init__(this,param):
        super(neko_cosconf_sim_NG, this).__init__();
        this.set_para(param);
    def partzreduce(this,partz_scr,conf):
        N,T,P,S=partz_scr.shape;
        if(P==1):
            return partz_scr.squeeze(2), conf.squeeze(2);
        diffw=(1-partz_scr);#focus on the most distinguish part,
        unsimw=trnf.softmax(diffw*trnf.sigmoid(conf),dim=2); # but less so if the model decide it does not capture the region well
        return (unsimw*partz_scr).sum(2),(conf).mean(2);

    def get_scr(this,dense_emb,protos):
        N, T, P, C = dense_emb.shape;
        conf=torch.clip(torch.norm(dense_emb,p=2,dim=-1,keepdim=True),0.00001); # to prevent nan
        fr=(dense_emb/conf).permute(2,0,1,3).reshape(P,N*T,C);
        fp=protos.permute(1,2,0); # P, M, NC
        pscr=fr.matmul(fp).permute(1,0,2).reshape(N,T,P,-1);
        oscr=this.partzreduce(pscr,conf);
        return oscr;

    # it has noting to do with plabel here.
    def forward(this, dense_emb, protos):

        cos,conf = this.get_scr(dense_emb, protos);

        # we look elsevier, but still treat it as a whole.
        return cos,conf;

class neko_2d_sp_cosconf_sim_NG(neko_cosconf_sim_NG):
    # it has noting to do with plabel here.
    def forward(this, dense_emb, protos):
        N, C, H, W = dense_emb.shape;
        cos,conf = this.get_scr(dense_emb.permute(0,2,3,1).view(N,H*W,1,C), protos);
        # we look elsevier, but still treat it as a whole.
        return cos.permute(0,2,1).view(N,-1,H,W),conf.permute(0,2,1).view(N,-1,H,W);
