import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# expect a list of
class neko_basic_lpred_attention_onemod(neko_module_wrapping_agent):
    INPUT_feats="feats";
    OUTPUT_lpred="lpred";
    OUTPUT_lpred_amax="lpred_amax";
    MOD_attmod="attm";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input(this.INPUT_feats,iocvt_dict);
        this.mod=this.register_mod(this.MOD_attmod,modcvt_dict);
        this.lpred=this.register_output(this.OUTPUT_lpred,iocvt_dict);
        this.amax=this.register_output(this.OUTPUT_lpred_amax,iocvt_dict);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        f=workspace.get(this.input);
        ff=f.reshape(f.shape[0],-1);
        lpred=environment(this.mod,ff );
        pred_length_amax=torch.argmax(lpred,-1)
        workspace.add(this.lpred, lpred);
        workspace.add(this.amax,pred_length_amax)
        return workspace;
def get_neko_basic_lpred_attention_onemod(feats_name,lpred_name,lpred_amax_name,attmod_name):
    engine = neko_basic_lpred_attention_onemod;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feats: feats_name, engine.OUTPUT_lpred: lpred_name, engine.OUTPUT_lpred_amax: lpred_amax_name}, "modcvt_dict": {engine.MOD_attmod: attmod_name}}}

if __name__ == '__main__':
    print(neko_basic_lpred_attention_onemod.get_default_configuration_scripts());