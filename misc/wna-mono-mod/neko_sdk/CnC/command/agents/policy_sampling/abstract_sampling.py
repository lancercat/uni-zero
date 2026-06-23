
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf
from neko_sdk.log import fatal

### we will leave this to the next week. The gist is it rabbits down
# Route everything to everyone. KISS (bcs we are rich now, lolol)
# just give actions, nothing else.

# this module aims keep sampling a constant number of action for each state.
# The users will be responsible to filter, if ever necessary
# remember: Cumulation will always happen after this.

class neko_abstract_policy_agent(neko_module_wrapping_agent):
    INPUT_states = "states"; # the states, not the united ones.
    INPUT_allow_mask="allow_mask";
    MOD_router = "router";

    OUTPUT_actions = "actions";  # what action actions has each data sample gone thru? [# sample,# path_per_sample]
    # if it takes more than one action to one states (e.g. ToT), give each path one an id.

    # the logits in the sense of all choices, ordered in the "original way".
    OUTPUT_logits="logits";
    OUTPUT_logp = "logp";

    def sample(this, logits,mask,log_probs):
        fatal("Not Impl")
        return torch.tensor(9);
        pass;


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.router = this.register_mod(this.MOD_router, modcvt_dict);
        this.states = this.register_input(this.INPUT_states, iocvt_dict);
        this.allow_mask = this.register_input(this.INPUT_allow_mask, iocvt_dict); # what is allowed to route to what?
        this.actions = this.register_output(this.OUTPUT_actions, iocvt_dict);
        this.logits=this.register_output(this.OUTPUT_logits,iocvt_dict);
        this.logp=this.register_output(this.OUTPUT_logp,iocvt_dict);


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        logits, _ = environment.module_dict[this.router](workspace.get(this.states));
        mask=workspace.get(this.allow_mask);
        if(len(logits.shape)>2):
            logits=logits.reshape([-1,logits.shape[-1]]); # logit[S,A]
        log_probs=trnf.log_softmax(logits,-1);
        actions=this.sample(logits,mask,log_probs);
        workspace.add(this.actions, actions);
        workspace.add(this.logits,logits);
        workspace.add(this.logp,log_probs);

        return workspace, environment;

