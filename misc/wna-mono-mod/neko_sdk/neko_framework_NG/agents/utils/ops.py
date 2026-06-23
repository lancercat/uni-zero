import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_sum_agent(neko_module_wrapping_agent):
    INPUT_srcs="inputs";
    OUTPUT_dst = "dst";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.srcs = this.register_input(this.INPUT_srcs, iocvt_dict);
        this.dst = this.register_output(this.OUTPUT_dst, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        allt=[workspace.get(src )for src in this.srcs];
        workspace.add(this.dst,torch.stack(allt,0).sum(0));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   srcs,
                   dst
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_srcs: srcs, cls.OUTPUT_dst: dst}, "modcvt_dict": {}}}

class neko_weighted_sum_agent(neko_module_wrapping_agent):
    INPUT_srcs="inputs";
    INPUT_weights="weights";
    OUTPUT_dst = "dst";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.srcs = this.register_input(this.INPUT_srcs, iocvt_dict);
        this.dst = this.register_output(this.OUTPUT_dst, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        allt=[workspace.get(src )for src in this.srcs];
        workspace.add(this.dst,torch.stack(allt,0).sum(0));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   srcs,
                   dst
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_srcs: srcs, cls.OUTPUT_dst: dst}, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_sum_agent.print_default_setup_scripts()