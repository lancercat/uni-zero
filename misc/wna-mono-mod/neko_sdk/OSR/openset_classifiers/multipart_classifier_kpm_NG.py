import torch

from neko_sdk.OSR.openset_classifiers.linear_classifier import neko_openset_linear_classifier
from neko_sdk.neko_score_merging import scatter_cvt

# okay this needs to be revised as well.
# unknown handling need to be decoupled from similarity computing.
# for now, we inject a unk to the plabel.
# S**T
class neko_openset_linear_classifierKPM(neko_openset_linear_classifier):
    def get_scr(this,flat_emb,protos):
        return flat_emb.permute(1,0,2).matmul(protos.permute(1,2,0));
    def get_unk_scr(this,flat_emb):
        return this.UNK_SCR*flat_emb.norm(dim=-1,keepdim=True);

    def forward(this, flat_emb, protos, plabel):
        if (len(flat_emb.shape) == 4):
            N, T, P, C = flat_emb.shape;
            t = this.forward(flat_emb.reshape(N * T, P, C), protos,plabel);
            return t.reshape(N, T, -1);
        out_res_ = torch.cat([this.get_scr(flat_emb, protos), this.get_unk_scr(flat_emb).permute(1,0,2)], -1);
        # we look elsevier, but still treat it as a whole.
        out_res=out_res_.mean(dim=0,keepdim=False)
        if (plabel is not None):
            scores = scatter_cvt(out_res, plabel);
        else:
            scores = out_res;
        return scores;