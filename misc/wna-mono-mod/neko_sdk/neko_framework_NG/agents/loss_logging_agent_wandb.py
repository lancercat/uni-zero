import os.path
import time
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.cfgtool.platform_cfg import platform_cfg, neko_platform_cfg


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_loss_logging_agent_wandb(neko_module_wrapping_agent):
    PARAM_prefix="prefix"
    def set_mod_io(this,iocvt_dict,modcvt_dict):
       pass;
    def set_etc(this,param):
        this.prefix=neko_get_arg(this.PARAM_prefix,param);
        this.time=time.time();

    # geez
    def flatten_dict_for_wandb(this,d,dd=None,prfx=""):
        if(dd is None):
            dd={};
        for k in d:
            if(type(k)!=str):
                continue; # well.
            if(type(d[k]) is dict):
                if(prfx==""):
                    prfxc=k;
                else:
                    prfxc=os.path.join(prfx,k);
                dd=this.flatten_dict_for_wandb(d[k],dd,prfxc);
            else:
                dd[k]=d[k];
        return dd;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        if(workspace.batch_idx and workspace.batch_idx%10==0):
            workspace.logdict["time"]=(time.time()-this.time)/10;
            this.run.log(this.flatten_dict_for_wandb(workspace.logdict,prfx=this.prefix));
            print(workspace.logdict);
            this.time = time.time();
        return workspace,environment;

def get_neko_logging_agent_wandb(cfg:neko_platform_cfg):
    agent=neko_loss_logging_agent_wandb
    return {
        "agent":agent,
        "params":{
            agent.PARAM_prefix:"training_log",
            "iocvt_dict":{
            },
            "modcvt_dict":
            {
            }
        }
    }