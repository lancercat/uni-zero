import time

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_loss_logging_agent(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
       pass;
    def set_etc(this,param):
        this.time=time.time();

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        if(workspace.batch_idx and workspace.batch_idx%10==0):
            workspace.logdict["time"]=(time.time()-this.time)/10;
            print(workspace.logdict)
            print(workspace.logdict);
            this.time = time.time();
        return workspace,environment;

def get_neko_logging_agent(cfg):
    return {
        "agent":neko_loss_logging_agent,
        "params":{
            "iocvt_dict":{
            },
            "modcvt_dict":
            {
            }
        }
    }