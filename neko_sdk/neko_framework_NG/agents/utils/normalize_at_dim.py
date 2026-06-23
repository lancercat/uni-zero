from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_simple_action_module_wrapping_agent_1i1o
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf

class neko_norm_at_agent(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_input="feat";
    OUTPUT_normed="normed";
    PARAM_normdim="norm_dim";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_input, iocvt_dict);
        this.normed = this.register_output(this.OUTPUT_normed, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.normdim = neko_get_arg(this.PARAM_normdim, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        input = workspace.get(this.input);
        workspace.add(this.normed,trnf.normalize(input, dim=this.normdim));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   input,
                   normed,
                   normdim
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_input: input, cls.OUTPUT_normed: normed},
                                         cls.PARAM_normdim: normdim, "modcvt_dict": {}}}
if __name__ == '__main__':
    neko_norm_at_agent.print_default_setup_scripts()