import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# average through a list of logits.
class neko_pre_joint_lpred_attention_onemod(neko_module_wrapping_agent):
    INPUT_feat_list="feats";
    OUTPUT_lpred="lpred";
    OUTPUT_lpred_amax="lpred_amax";
    MOD_attmod="attm";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.inputs=this.register_input_list(this.INPUT_feat_list,iocvt_dict);
        this.lpred=this.register_output(this.OUTPUT_lpred,iocvt_dict);
        this.amax=this.register_output(this.OUTPUT_lpred_amax,iocvt_dict);
        this.mod=this.register_mod(this.MOD_attmod,modcvt_dict);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fs=[];
        for fn in this.inputs:
            f=workspace.get(fn);
            fs.append(f.reshape(f.shape[0], -1));
        jf=torch.cat(fs,dim=-1);
        lpred=environment(this.mod,jf );
        pred_length_amax=torch.argmax(lpred,-1)
        workspace.add(this.lpred, lpred);
        workspace.add(this.amax,pred_length_amax)
        return workspace;
def get_neko_pre_joint_lpred_attention_onemod(feat_list,lpred,lpred_amax,attmod):
    engine = neko_pre_joint_lpred_attention_onemod;
    return {
        "agent": engine,
        "params": {
            "iocvt_dict": {
                engine.INPUT_feat_list: feat_list, engine.OUTPUT_lpred: lpred, engine.OUTPUT_lpred_amax: lpred_amax}, "modcvt_dict": {engine.MOD_attmod: attmod}
        }
    }

if __name__ == '__main__':
    print(neko_pre_joint_lpred_attention_onemod.print_default_setup_scripts());