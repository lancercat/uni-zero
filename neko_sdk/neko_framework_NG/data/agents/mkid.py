import torch

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
class neko_id_maker_agent(ama):
    INPUT_samples="samples";
    OUTPUT_ids="ids";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.samples = this.register_input(this.INPUT_samples, iocvt_dict);
        this.ids = this.register_output(this.OUTPUT_ids, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        samples = workspace.get(this.samples);
        workspace.add(this.ids,torch.arange(len(samples),dtype=torch.int));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   samples,
                   ids
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_samples: samples, cls.OUTPUT_ids: ids}, "modcvt_dict": {}}}



if __name__ == '__main__':
    neko_id_maker_agent.print_default_setup_scripts();
