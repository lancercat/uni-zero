from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred,neko_flatten_NG_idx_mapping
import torch


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
# in mk4 it just samples. responsible for length control is shifted to far above this level---
# if you want shorter stuff, produce less attention maps :)
class neko_word_aggr_mk4(neko_module_wrapping_agent):
    INPUT_feature_name="feature_name";
    INPUT_attention_map_name = "attention_map_name";
    OUTPUT_feat_seq_name="full_feat_seq_name";
    MOD_seq="seq_mod_name";

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.feat=this.register_input(this.INPUT_feature_name,iocvt_dict);
        this.att = this.register_input(this.INPUT_attention_map_name, iocvt_dict);
        this.feat_seq=this.register_output(this.OUTPUT_feat_seq_name,iocvt_dict);
        this.seq_mod=this.register_mod(this.MOD_seq,modcvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        out_emb = environment.module_dict[this.seq_mod](
            workspace.get(this.feat),
            workspace.get(this.att));
        workspace.add(this.feat_seq,out_emb);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                    feature_name,attention_map_name,
                   feat_seq_name,
                   seq
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_attention_map_name: attention_map_name, cls.INPUT_feature_name: feature_name,
                           cls.OUTPUT_feat_seq_name: feat_seq_name}, "modcvt_dict": {cls.MOD_seq: seq}}}


if __name__ == '__main__':
    neko_word_aggr_mk4.print_default_setup_scripts();
