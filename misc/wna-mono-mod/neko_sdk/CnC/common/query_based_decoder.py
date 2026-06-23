
import torch
from torch import nn

from neko_sdk.seq2seq.neko_fixed_torch_transformer import neko_TransformerEncoderLayer
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se


class neko_command_self_att_fn(nn.Module):
    def __init__(this, featdim,num_head=3,num_l=3):
        super().__init__();
        this.sa = nn.TransformerEncoder(neko_TransformerEncoderLayer(featdim, num_head), num_l);

    def forward(this, feat,queries):
        nB,nC=feat.shape[:2];
        return this.sa(torch.cat([queries, feat.reshape(nB,nC, -1).permute(0,2,1)],dim=1))[:, :queries.shape[1]];

class neko_command_se_self_att_fn(nn.Module):
    def __init__(this, featdim, se_dim=32,num_head=3,num_l=3):
        super().__init__();
        this.se = neko_add_embint_se(c=se_dim);
        this.sa = nn.TransformerEncoder(neko_TransformerEncoderLayer(featdim, num_head), num_l);

    def forward(this, feat,queries):
        nB=feat.shape[0];
        fmp = this.se(feat).reshape(nB, -1);
        return this.sa(torch.cat([queries, fmp]))[:, :queries.shape[1]];
