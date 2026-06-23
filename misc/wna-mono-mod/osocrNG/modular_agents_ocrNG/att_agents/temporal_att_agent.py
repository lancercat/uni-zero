import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent,neko_simple_action_module_wrapping_agent_1i1o
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# expect a list of
class neko_basic_temporal_attention(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_feats="feats";
    OUTPUT_masks="amsks";
    MOD_attmod="attm";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input(this.INPUT_feats,iocvt_dict);
        this.mod=this.register_mod(this.MOD_attmod,modcvt_dict);
        this.output=this.register_output(this.OUTPUT_masks,iocvt_dict);
    @classmethod
    def get_agtcfg(cls,feats_name,
        masks_name,
        attmod_name):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_feats: feats_name, cls.OUTPUT_masks: masks_name},
                           "modcvt_dict": {cls.MOD_attmod: attmod_name}}}


def get_neko_basic_temporal_attention(feats_name,
masks_name,
attmod_name):
    engine = neko_basic_temporal_attention;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name, engine.OUTPUT_masks: masks_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}
class neko_basic_attention_lastfeat(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_feats="feats";
    OUTPUT_masks="amsks";
    MOD_attmod="attm";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input(this.INPUT_feats,iocvt_dict);
        this.mod=this.register_mod(this.MOD_attmod,modcvt_dict);
        this.output=this.register_output(this.OUTPUT_masks,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        workspace.add(this.output, environment(this.mod, workspace.get(this.input)[-1]));
        return workspace;

    @classmethod
    def get_agtcfg(cls,
                   feats,
                   masks,
                   attmod
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_feats: feats, cls.OUTPUT_masks: masks},
                                         "modcvt_dict": {cls.MOD_attmod: attmod}}}


class neko_basic_temporal_attention_combined(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_feats = "feats";
    INPUT_another = "another";

    OUTPUT_masks = "amsks";
    MOD_attmod = "attm";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_feats, iocvt_dict);
        this.another=this.register_input(this.INPUT_another,iocvt_dict);
        this.mod = this.register_mod(this.MOD_attmod, modcvt_dict);
        this.output = this.register_output(this.OUTPUT_masks, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        f=torch.cat([workspace.get(this.input),workspace.get(this.another)],dim=1);
        workspace.add(this.output, environment(this.mod,f ));
        return workspace;

class neko_basic_attention_lastfeat_och_others(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_feats = "feats";
    INPUT_another = "another";

    OUTPUT_masks = "amsks";
    MOD_attmod = "attm";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_feats, iocvt_dict);
        this.another=this.register_input(this.INPUT_another,iocvt_dict);
        this.mod = this.register_mod(this.MOD_attmod, modcvt_dict);
        this.output = this.register_output(this.OUTPUT_masks, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        f=torch.cat([workspace.get(this.input)[-1],workspace.get(this.another)],dim=1);
        workspace.add(this.output, environment(this.mod,f ));
        return workspace;


def get_neko_basic_temporal_attention(feats_name,
masks_name,
attmod_name):
    engine = neko_basic_temporal_attention;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name, engine.OUTPUT_masks: masks_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}

def get_neko_basic_attention_lastfeat(feats_name,
masks_name,
attmod_name):
    engine = neko_basic_attention_lastfeat;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name, engine.OUTPUT_masks: masks_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}
def get_neko_basic_temporal_attention_combined(feats_name,another_name,
masks_name,
attmod_name):
    engine = neko_basic_temporal_attention_combined;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name,engine.INPUT_another:another_name, engine.OUTPUT_masks: masks_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}

def get_neko_basic_attention_lastfeat_och_another(feats_name,another_name,
masks_name,
attmod_name):
    engine = neko_basic_attention_lastfeat_och_others;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name,engine.INPUT_another:another_name, engine.OUTPUT_masks: masks_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}

if __name__ == '__main__':
    neko_basic_attention_lastfeat.print_default_setup_scripts();