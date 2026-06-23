import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class avging_loss_agent_mk2(neko_module_wrapping_agent):
    INPUT_per_item_loss_name = "per_instance_loss_name";
    OUTPUT_loss_name = "loss_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.item_loss=this.register_input(this.INPUT_per_item_loss_name,iocvt_dict);
        this.lossname=this.register_output(this.OUTPUT_loss_name,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_loss =workspace.get(this.item_loss);
        loss=per_inst_loss.mean();
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname]=loss.item();
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   per_item_loss_name,
                   loss_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_per_item_loss_name: per_item_loss_name, cls.OUTPUT_loss_name: loss_name},
            "modcvt_dict": {}}}


def get_avging_loss_agent_mk2(per_item_loss_name,
                              loss_name
                              ):
    engine = avging_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_per_item_loss_name: per_item_loss_name,
                                                       engine.OUTPUT_loss_name: loss_name}, "modcvt_dict": {}}}


if __name__ == '__main__':
    avging_loss_agent_mk2.print_default_setup_scripts()