
from torch.nn import functional as trnf
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import torch
class abstract_aggr(ama):
    MOD_drop="drop";
    INPUT_featmap="featmap";
    INPUT_attnmap="attnmaps";
    OUTPUT_feat_vec="featvecs";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.featmap = this.register_input(this.INPUT_featmap, iocvt_dict);
        this.attnmaps = this.register_input(this.INPUT_attnmap, iocvt_dict);
        this.feat_vec = this.register_output(this.OUTPUT_feat_vec, iocvt_dict);
        this.drop = this.register_mod(this.MOD_drop, modcvt_dict,"NEP_skipped_NEP");
        pass;

    def set_etc(this, params):
        pass;


    def attn_core(this, workspace: neko_workspace, environment: neko_environment):


        return out_emb
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        featmap = workspace.get(this.featmap);
        N, C, H, W = featmap.shape;
        A_ = workspace.get(this.attnmaps);

        if (H != A_.shape[-2] or W != A_.shape[-1]):
            A = trnf.interpolate(A_, (H, W));
        else:
            A = A_;
        out_emb=this.do_attn(featmap,A)
        if (this.drop is not None):
            out_emb = environment.module_dict[this.drop](out_emb);
        workspace.add(this.feat_vec,out_emb.squeeze(-1));

        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
        attnmap,featmap,
        feat_vec,
        drop
    ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_attnmap: attnmap, cls.INPUT_featmap: featmap, cls.OUTPUT_feat_vec: feat_vec}, "modcvt_dict": {cls.MOD_drop: drop}}}

# agentic implementation.
# the gist is bogo model will be exclusively used for backbone.
class aggrseq(abstract_aggr):

    def do_attn(this,featmap,A):
        fatal("notimpl")

class aggrone(abstract_aggr):
    def do_attn(this,featmap,A):
        return (A.unsqueeze(2) * featmap.unsqueeze(1)).sum(-1).sum(-1)  /torch.clip(A.sum(-1).sum(-1).unsqueeze(2),0.00000009);


if __name__ == '__main__':
    aggrone.print_default_setup_scripts()
