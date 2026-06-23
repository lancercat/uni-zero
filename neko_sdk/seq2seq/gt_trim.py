from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama



class neko_list_trim_maxT_alot_agent(ama):
    INPUT_SRCs = "srcs";
    OUTPUT_targets = "targets";
    PARAM_maxT="maxT";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.srcs = this.register_input_list(this.INPUT_SRCs, iocvt_dict);
        this.aggrtars = this.register_output_list(this.OUTPUT_targets, iocvt_dict);
    def set_etc(this,param):
        this.maxL=neko_get_arg(this.PARAM_maxT,param)-1;
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for src,tar in zip(this.srcs,this.aggrtars):
            workspace.add(tar,[si[:this.maxL] for si in workspace.get(src)]);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   srcs,
                   targets,
                   maxT
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_SRCs: srcs, cls.OUTPUT_targets: targets}, "modcvt_dict": {},cls.PARAM_maxT:maxT}}
