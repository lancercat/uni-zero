import os.path
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

# select based on priority.

class neko_rule_based_priority_agent(ama):
    INPUT_in_pri_list="prilst";
    OUTPUT_out_target = "target";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.in_pri_list = this.register_input(this.INPUT_in_pri_list, iocvt_dict);
        this.out_target = this.register_output(this.OUTPUT_out_target, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for n in this.in_pri_list:
            if n in workspace.inter_dict:
                workspace.alias(n,this.out_target);
                break;
            else:
                pass;

        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   in_pri_list,
                   out_target
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_in_pri_list: in_pri_list, cls.OUTPUT_out_target: out_target},
                           "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_rule_based_priority_agent.print_default_setup_scripts();
