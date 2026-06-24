import torch
from torch.nn import functional as trnf
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# expect a list of
class neko_basic_attention_selection(neko_module_wrapping_agent):
    INPUT_feats="feats";
    OUTPUT_attsel="lpred";
    MOD_selmod="selm";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input(this.INPUT_feats,iocvt_dict);
        this.mod=this.register_mod(this.MOD_selmod,modcvt_dict);
        this.sel=this.register_output(this.OUTPUT_attsel,iocvt_dict);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        f=workspace.get(this.input);
        ff=f.reshape(f.shape[0],-1);
        sel=environment(this.mod,ff );
        workspace.add(this.sel, sel);
        return workspace;
def get_neko_basic_attention_selection(
    feats,
    attsel,
    selmod
):
    engine = neko_basic_attention_selection;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats, engine.OUTPUT_attsel: attsel}, "modcvt_dict": {engine.MOD_selmod: selmod}}}


class neko_basic_attention_selection_mk2(neko_module_wrapping_agent):
    INPUT_feats="feats";
    OUTPUT_attsel_prob="attselprob";
    OUTPUT_attsel_logit="attsellogit"
    MOD_selmod="selm";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feats = this.register_input(this.INPUT_feats, iocvt_dict);
        this.attsel_logit = this.register_output(this.OUTPUT_attsel_logit, iocvt_dict);
        this.attsel_prob = this.register_output(this.OUTPUT_attsel_prob, iocvt_dict);
        this.selmod = this.register_mod(this.MOD_selmod, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        feats = workspace.get(this.feats);
        logits=environment(this.selmod,feats.reshape(feats.shape[0],-1));
        probs=trnf.softmax(logits,-1);
        workspace.add(this.attsel_prob,probs);
        workspace.add(this.attsel_logit,logits);
        return workspace, environment;

def get_neko_basic_attention_selection_mk2(
        feats,
        attsel_logit, attsel_prob,
        selmod
):
    engine = neko_basic_attention_selection_mk2;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_feats: feats, engine.OUTPUT_attsel_logit: attsel_logit,
                       engine.OUTPUT_attsel_prob: attsel_prob}, "modcvt_dict": {engine.MOD_selmod: selmod}}}


if __name__ == '__main__':
    neko_basic_attention_selection_mk2.print_default_setup_scripts();