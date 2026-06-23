# logits:  *, #classes
# utfs: * (type of string)
# how to reorganize the utfs are up to tasks
import numpy as np
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
class neko_point_wise_translation(ama):
    INPUT_logits="logits";
    INPUT_tdict="tdict";
    OUTPUT_utfs="utfs";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.logits = this.register_input(this.INPUT_logits, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.utfs = this.register_output(this.OUTPUT_utfs, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;
    def translate(this,logits,tdict):
        max_cls=torch.argmax(logits.view([-1,logits.shape[-1]]),dim=-1).cpu().numpy();
        futf=np.array([tdict[i] for i in max_cls]);
        return futf.reshape(logits.shape[:-1]);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        logits = workspace.get(this.logits);
        tdict = workspace.get(this.tdict);
        workspace.add(this.utfs,this.translate(logits,tdict));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   logits, tdict,
                   utfs
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_logits: logits, cls.INPUT_tdict: tdict, cls.OUTPUT_utfs: utfs},
                           "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_point_wise_translation.print_default_setup_scripts()
