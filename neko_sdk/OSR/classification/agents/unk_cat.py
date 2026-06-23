import torch

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama


class neko_inject_cos_thresh_agent(ama):
    INPUT_in_cossim="in_cossim";
    OUTPUT_out_cossim="out_cossim";
    MOD_thresholder="thresholder"

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.in_cossim = this.register_input(this.INPUT_in_cossim, iocvt_dict);
        this.out_cossim = this.register_output(this.OUTPUT_out_cossim, iocvt_dict);
        this.thresholder = this.register_mod(this.MOD_thresholder, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        in_cossim = workspace.get(this.in_cossim);
        th=environment(this.thresholder)+torch.zeros_like(in_cossim[:,:,-1:]);
        workspace.add(this.out_cossim,torch.cat([in_cossim,th],-1));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   in_cossim,
                   out_cossim,
                   thresholder
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_in_cossim: in_cossim, cls.OUTPUT_out_cossim: out_cossim},
                           "modcvt_dict": {cls.MOD_thresholder: thresholder}}}



class neko_inject_cos_thresh_2d_agent(neko_inject_cos_thresh_agent):



    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        in_cossim = workspace.get(this.in_cossim);
        th=environment(this.thresholder)+torch.zeros_like(in_cossim[:,-1:]);
        workspace.add(this.out_cossim,torch.cat([in_cossim,th],1));
        return workspace, environment;

if __name__ == '__main__':
    neko_inject_cos_thresh_agent.print_default_setup_scripts();
