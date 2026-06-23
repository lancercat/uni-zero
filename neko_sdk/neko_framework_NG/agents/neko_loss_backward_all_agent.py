import os

import torch

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace,neko_environment
# which sums all losses in objdict and commence backward function.
from neko_sdk.log import fatal
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.debug import panic
# this is to prepare for separation different bp flows for async training.
# ---- if some of them share the workspace somehow

class neko_basic_backward_all_core(neko_abstract_sync_agent):

    def take_action_core(this,workspace:neko_workspace,environment:neko_environment):
        if(len(workspace.objdict)==0):
            return torch.tensor(0); # for somereason, this fpbp round skips this task.
        tl=0;
        for l in workspace.objdict:
            tl=tl+workspace.objdict[l];
            if tl.isnan():
                panic(workspace,environment);
                fatal("error!!!!"+l+"IS NAN!!");
        return tl;

class neko_basic_backward_all_agent(neko_basic_backward_all_core):
    
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        tl=this.take_action_core(workspace,environment);
        # logging is nomore managed by backward agent,
        # we drop the losses that has been back propagated once,
        # so they do not get back propagated for a second time....
        # well if you want more complexity control override this one...
        if(tl>0.000000009):
            tl.backward();
        workspace.objdict={};
        return workspace,environment;

def get_neko_basic_backward_all_agent():
    return {
        "agent":neko_basic_backward_all_agent,
        "params":{}
    }


class neko_basic_backward_all_agent_logged(neko_basic_backward_all_core):
    PARAM_total_backward_name="total_backward_name";
    def setup(this,param):
        this.logname=neko_get_arg(this.PARAM_total_backward_name,param,"NEP_skipped_NEP");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        tl=this.take_action_core(workspace,environment);
        # logging is nomore managed by backward agent,
        # we drop the losses that has been back propagated once,
        # so they do not get back propagated for a second time....
        # well if you want more complexity control override this one...
        if(this.logname is not None):
            workspace.add_log(this.logname,tl.item());
        tl.backward();
        workspace.objdict={};
        return workspace,environment;

