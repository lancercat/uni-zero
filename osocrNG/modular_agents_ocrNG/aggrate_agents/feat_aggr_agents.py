from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred,neko_flatten_NG_idx_mapping
import torch

# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_word_aggr(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.feature=neko_get_arg("feature_name",iocvt_dict);
        this.input_dict.length=neko_get_arg("length_name",iocvt_dict);
        this.input_dict.attmap=neko_get_arg("attention_map_name",iocvt_dict);
        this.output_dict.feat_seq=neko_get_arg("feat_seq_name",iocvt_dict);
        this.mnames.seq_name=neko_get_arg("seq_mod_name",modcvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        out_emb = environment.module_dict[this.mnames.seq_name](
            workspace.inter_dict[this.input_dict.feature][-1],
            workspace.inter_dict[this.input_dict.attmap],
            workspace.inter_dict[this.input_dict.length]);
        fout_emb, _ = neko_flatten_NG_lenpred.inflate(out_emb, workspace.inter_dict[this.input_dict.length]);
        workspace.inter_dict[this.output_dict.feat_seq]=fout_emb;
        return workspace,environment;

# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_word_aggr_mk2(neko_module_wrapping_agent):
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
            workspace.get(this.feat)[-1],
            workspace.get(this.att),
            None);
        # this will cause it to aggregate all features. But that shan't be of much computation here.
        # one stamp worths wh(~768)*c(~512) so there is not much to worry abt compared to feature computations.
        workspace.add(this.feat_seq,out_emb);
        return workspace,environment;

class neko_word_aggr_mk2l(neko_word_aggr_mk2):
    INPUT_length_name="length_name";
    PARAM_maxT="maxT";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.len=this.register_input(this.INPUT_length_name,iocvt_dict);
    def set_etc(this,param):
        this.maxT=neko_get_arg(this.PARAM_maxT,param,9999);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        l=workspace.get(this.len)+0; # get a copy.
        att=workspace.get(this.att);
        l=torch.clip(l,max=att.shape[1]); # currently, we don't stop commander routing too long stuff. Will be fixed in near future.

        out_emb = environment.module_dict[this.seq_mod](
            workspace.get(this.feat)[-1],
            att,l
            );
        # this will cause it to aggregate all features. But that shan't be of much computation here.
        # one stamp worths wh(~768)*c(~512) so there is not much to worry abt compared to feature computations.
        workspace.add(this.feat_seq,out_emb);
        return workspace,environment;
class neko_word_aggr_mk3(neko_word_aggr_mk2):
    PARAM_maxT="maxT";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
    def set_etc(this,param):
        this.maxT=neko_get_arg(this.PARAM_maxT,param,9999);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        l=workspace.get(this.len)+0; # get a copy.
        att=workspace.get(this.att);
        l=torch.clip(l,max=att.shape[1]); # currently, we don't stop commander routing too long stuff. Will be fixed in near future.

        out_emb = environment.module_dict[this.seq_mod](
            workspace.get(this.feat)[-1],
            att,l
            );
        # this will cause it to aggregate all features. But that shan't be of much computation here.
        # one stamp worths wh(~768)*c(~512) so there is not much to worry abt compared to feature computations.
        workspace.add(this.feat_seq,out_emb);
        return workspace,environment;
class neko_word_aggr_mk2ll(neko_word_aggr_mk2):
    INPUT_length_name="length_name";
    PARAM_maxT="maxT";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.len=this.register_input(this.INPUT_length_name,iocvt_dict);
    def set_etc(this,param):
        this.maxT=neko_get_arg(this.PARAM_maxT,param,9999);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        l=workspace.get(this.len)+0; # get a copy.
        att=workspace.get(this.att);
        l=torch.clip(l,max=att.shape[1]); # currently, we don't stop commander routing too long stuff. Will be fixed in near future.

        out_emb = environment.module_dict[this.seq_mod](
            workspace.get(this.feat)[-1],
            att,l
            );
        # this will cause it to aggregate all features. But that shan't be of much computation here.
        # one stamp worths wh(~768)*c(~512) so there is not much to worry abt compared to feature computations.
        workspace.add(this.feat_seq,out_emb);
        return workspace,environment;


class neko_word_flatten_agent(neko_module_wrapping_agent):
    INPUT_feat_seq_name="feat_seq_name";
    INPUT_length_name="length_name";
    OUTPUT_flatten_feat_seq_name="flatten_feat_seq_name";
    OUTPUT_flatten_mapping_name="flatten_mapping";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat_seq=this.register_input(this.INPUT_feat_seq_name,iocvt_dict);
        this.len=this.register_input(this.INPUT_length_name,iocvt_dict);
        this.feat_seq_flatten=this.register_output(this.OUTPUT_flatten_feat_seq_name,iocvt_dict);
        this.mapping=this.register_output(this.OUTPUT_flatten_mapping_name,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        # if we don't know for sure how long it is, we guess.
        fout_emb,id  = neko_flatten_NG_idx_mapping.inflate(workspace.get(this.feat_seq), workspace.get(this.len));
        workspace.add(this.feat_seq_flatten,fout_emb);
        workspace.add(this.mapping,id);
        return workspace,environment;