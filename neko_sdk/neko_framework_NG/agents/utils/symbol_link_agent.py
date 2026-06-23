# assumes we have one single agent for single input
# the agent will pick RANDOMLY if there are more than one agents assigned to the same input.
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

# just one single element, so do nothing.
# This happens when
class neko_symbol_link_agent(neko_module_wrapping_agent):
    INPUT_SRC="src";
    OUTPUT_target="target";

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.src=this.register_input_list(this.INPUT_SRC,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        SRC = workspace.get(this.src);
        workspace.add(this.aggrtar,SRC);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   src,
                   target
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_SRC: src, cls.OUTPUT_target: target}, "modcvt_dict": {}}}

def get_neko_symbol_link_agent(
    src,
    target
):
    engine = neko_symbol_link_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_SRC: src, engine.OUTPUT_target: target}, "modcvt_dict": {}}}

class neko_symbol_link_alot_agent(neko_module_wrapping_agent):
    INPUT_SRCs = "srcs";
    OUTPUT_targets = "targets";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.srcs = this.register_input_list(this.INPUT_SRCs, iocvt_dict);
        this.aggrtars = this.register_output(this.OUTPUT_targets, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for src,tar in zip(this.srcs,this.aggrtars):
            workspace.add(tar, workspace.get(src) );
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   srcs,
                   targets
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_SRCs: srcs, cls.OUTPUT_targets: targets}, "modcvt_dict": {}}}
class neko_symbol_link_alot_agent_weak(neko_symbol_link_alot_agent):
    # sometimes not all srcs will present due to routing sampler and other behaviour.
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for src,tar in zip(this.srcs,this.aggrtars):
            if(src in workspace.inter_dict):
                workspace.add(tar, workspace.get(src) );
        return workspace, environment;