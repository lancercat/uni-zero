from neko_sdk.neko_framework_NG.bogog2_modules.neko_abstract_bogo_g2 import neko_abstract_bogo_g2
class gen4_object_to_feat_abstract(neko_abstract_bogo_g2):
    def fe(this,clips):
        features = this.backbone(clips)
        A = this.aggr(features);
        # A=torch.ones_like(A);
        out_emb=(A.unsqueeze(2)*features[-1].unsqueeze(1)).sum(-1).sum(-1)/A.unsqueeze(2).sum(-1).sum(-1);
        return out_emb;

    def fe_debug(this,clips):
        features = this.backbone(clips);
        A = this.aggr(features);
        # A=torch.ones_like(A);
        out_emb=(A*features[-1]).sum(-1).sum(-1)/A.sum(-1).sum(-1);
        return out_emb,A;
    def declare_mods(this):
        this.aggr=None;
        this.backbone=None;
        this.drop=None;
    def check_cvt(this,mod_cvt_dict):
        mod_cvt_dict["backbone"];
        mod_cvt_dict["aggr"];
        return True;
    def forward(this, normprotos):
        raw_protos=this.fe(normprotos);
        if ("drop" in this.modnames):
            raw_protos = this.drop(raw_protos);
        return raw_protos;


