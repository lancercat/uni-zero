import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
# Filters applied pre_routing, can be chained
class neko_pre_filter_agent(neko_module_wrapping_agent):
    OUTPUT_restrict_mask_out="outmask";
    INPUT_restrict_mask_in="restrict_mask";
    INPUT_raw_ims="raw_ims";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.inpm = this.register_input(this.INPUT_restrict_mask_in, iocvt_dict);
        this.oupm = this.register_output(this.OUTPUT_restrict_mask_out, iocvt_dict);
        this.raw_ims=this.register_input(this.INPUT_raw_ims,iocvt_dict);
    def set_etc(this,param):
        this.acnt=0;
        fatal("you need to setup acnt in your filter class")
    def filter(this,raw_im,basemsk,environment,workspace):
       pass;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        raw_ims=workspace.get(this.raw_ims);
        with torch.no_grad():
            if (this.inpm is not None):
                out = workspace.get(this.inpm) + 0; # copy
            else:
                out = torch.ones((len(raw_ims),this.acnt));
            this.filter(raw_ims,out,workspace,environment);

        workspace.add(this.oupm,out);
        return workspace,environment