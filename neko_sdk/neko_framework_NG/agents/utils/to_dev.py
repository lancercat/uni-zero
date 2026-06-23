
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

import torch

class neko_to_dev(ama):
    INPUT_in_dev_reference="in_dev_ref"; # just make sure 0th is a tensor with device method
    INPUT_in_things="in_things";
    OUTPUT_out_things="out_things";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.in_dev_reference = this.register_input(this.INPUT_in_dev_reference, iocvt_dict);
        this.in_things = this.register_input(this.INPUT_in_things, iocvt_dict);
        this.out_things = this.register_output(this.OUTPUT_out_things, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;
    def to(this,listortensor,device):
        if(type(listortensor) == torch.Tensor):
            return listortensor.to(device);
        else:
            return [this.to(i,device) for i in listortensor];

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if (this.in_dev_reference not in workspace.inter_dict):
            device = workspace.device;
        else:
            in_dev_reference = workspace.get(this.in_dev_reference);
            device = in_dev_reference[0].device;
        in_things = [workspace.get(i) for i in  this.in_things];
        for i,on in zip(in_things,this.out_things):
            workspace.add(on,this.to(i,device),True); # if we just move things to device then don't sweat the details.
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   in_dev_reference, in_things,
                   out_things
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_in_dev_reference: in_dev_reference, cls.INPUT_in_things: in_things,
                           cls.OUTPUT_out_things: out_things}, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_to_dev.print_default_setup_scripts();
