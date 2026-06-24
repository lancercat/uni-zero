from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.io.encdec import decode_prob;
from osocrNG.names import default_ocr_variable_names as dvn
from neko_sdk.log import warn
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
from neko_sdk.ocr_modules.sptokens import tUNKREP

import torch
class simple_pred_agent(neko_module_wrapping_agent):
    MOD_pred_name="pred_name";
    INPUT_feat_seq_name=dvn.feat_seq_name;
    INPUT_tensor_proto_vec_name=dvn.tensor_proto_vec_name;
    INPUT_proto_label_name=dvn.proto_label_name;
    OUTPUT_logit_name=dvn.logit_name;

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred = this.register(this.MOD_pred_name,modcvt_dict,this.mnames);

        this.feat_seq_name=this.register(this.INPUT_feat_seq_name,iocvt_dict,this.input_dict);
        this.tensor_proto_vec_name=this.register(this.INPUT_tensor_proto_vec_name,iocvt_dict,this.input_dict);

        this.proto_label_name=this.register(this.INPUT_proto_label_name,iocvt_dict,this.input_dict);
        this.logit_name=this.register(this.OUTPUT_logit_name,iocvt_dict,this.output_dict);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        workspace.inter_dict[this.logit_name] = environment.module_dict[this.pred](
            workspace.inter_dict[this.feat_seq_name],
            workspace.inter_dict[this.tensor_proto_vec_name],
            workspace.inter_dict[this.proto_label_name]
        );
        return workspace;

# it has no state(unlike loggers with histories), so it does not have a module.
class translate_agent(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.logit_name=neko_get_arg(dvn.logit_name,iocvt_dict);
        this.input_dict.tdict_name=neko_get_arg(dvn.tdict_name,iocvt_dict);
        this.input_dict.length_name=neko_get_arg(dvn.length_name,iocvt_dict);
        this.output_dict.pred_text_name=neko_get_arg(dvn.pred_text_name,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        try:
            outpred=decode_prob(
                workspace.inter_dict[this.input_dict.logit_name],
                workspace.inter_dict[this.input_dict.length_name],
                workspace.inter_dict[this.input_dict.tdict_name]
            )[0];
        except:
            warn("badpred");
            # print(workspace.inter_dict);
            outpred=["BADPRED" for i in range(workspace.inter_dict[this.input_dict.length_name].shape[0])];
        workspace.inter_dict[this.output_dict.pred_text_name]=outpred;
        pass;

class translate_gt_agent(neko_module_wrapping_agent):
    INPUT_gt_text="GT";
    INPUT_tdict="tdict";
    OUTPUT_gt_text_wunk= "GT_wunk"; # damn, if we use un-unked text to compute reward we are nuking ourselves.

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.gt_text = this.register_input(this.INPUT_gt_text, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.gt_text_wunk = this.register_output(this.OUTPUT_gt_text_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt_text = workspace.get(this.gt_text);
        tdict = workspace.get(this.tdict);
        wunk=["".join([ c if c in tdict else tUNKREP for c in tokenize(gt)])for gt in gt_text];
        workspace.add(this.gt_text_wunk,wunk);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   gt_text, tdict,
                   gt_text_wunk
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_gt_text: gt_text, cls.INPUT_tdict: tdict, cls.OUTPUT_gt_text_wunk: gt_text_wunk},
            "modcvt_dict": {}}}

def get_translate_gt_agent(
        gt_text, tdict,
        gt_text_wunk
):
    engine = translate_gt_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_gt_text: gt_text, engine.INPUT_tdict: tdict,
                                                       engine.OUTPUT_gt_text_wunk: gt_text_wunk},
                                        "modcvt_dict": {}}}


def get_pred_agent(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,
                   pred_name):
    return {
        "agent":simple_pred_agent,
        "params":{
            "iocvt_dict":{
                "feat_seq_name":feat_seq_name,
                "tensor_proto_vec_name":tensor_proto_vec_name,
                "proto_label_name":proto_label_name,
                "logit_name":logit_name
            },
            "modcvt_dict":{
                "pred_name":pred_name,
            }
        }
    }
def get_translate_agent(logit_name,tdict_name,length_name,pred_text_name):
    return {
        "agent":translate_agent,
        "params":{
            "iocvt_dict":{
              "logit_name":logit_name,
              "tdict_name":tdict_name,
              "length_name":length_name,
              "pred_text_name":pred_text_name,
            },
            "modcvt_dict": {
            }
        }
    }

if __name__ == '__main__':
    translate_gt_agent.print_default_setup_scripts();
