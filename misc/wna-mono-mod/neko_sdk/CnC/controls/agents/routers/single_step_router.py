# we will think about how to do it more elegantly in the future,
# but now it is just a single step routing based on names
# (so we don't have to keep a tab on what went where with what probability).
# ROADMAP
#  [Chains]->Tree->DAG->Free form routing-> Dynamic & self-describing nodes.
#     ^----- we are here

# It's okay to "waste" some computation resources bcs we have berzerlius!
# Let's sample with some more aggressive policies!

#
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from torch.distributions import Categorical
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf

# if there are more than one action and you believe we shouldn't just rerun it, we will add a new class.
# will be more efficient to do a torch scatter and then slice. but alas there are lists \(-_-)/
# we will in the future at least make this agent cpu parallel....


# this will not allow you to dynamically add agents (extensions)
# but heck we are not doing this in 2024 anyway.

# this module carries the data to the policy
class neko_single_step_name_based_routing_agent_static(neko_module_wrapping_agent):
    PARAM_routing_subjects="routing_subjects";
    PARAM_prefix="prefix";
    PARAM_names="names"; # the names of each path; element in single step
    INPUT_actions="actions"; # what action paths has each data sample gone thru? [# sample,# path_per_sample] list of  element in single step
    INPUT_raw_log_prob="in_log_prob"; # the log prob of each selected path.


    # if it takes more than one action to one states (e.g. ToT), give each path one an id.
    OUTPUT_weightath_log_prob="out_log_prob"; # the routed logprob
    OUTPUT_weightath_ids="path_ids"; # Which path did/shall the data follow
    OUTPUT_weightath_detached_log_prob="detached_log_prob";
    OUTPUT_sample_id="samid"; # the id of each sample
    #  Well if you have the full distribution you don't have to monte carlo thru.
    # PARAM_sample_number="sample_number"; # Set it to the Histy number (3) pls.
    # PARAM_sample_topk = "sample_topk";  # Only sample from top k options, Set it to the Histy number (3) pls.
    def tarname(this,name,prfx):
        return this.prefix+prfx + name.replace(this.prefix, "");

    def set_etc(this,param):
        # this.sample_number=neko_get_arg(this.PARAM_sample_number,param,3);
        this.routing_subjects=neko_get_arg(this.PARAM_routing_subjects,param);
        this.prefix=neko_get_arg(this.PARAM_prefix,param,"");
        this.names=neko_get_arg(this.PARAM_names,param);
        for n in this.names:
            for k in this.routing_subjects:
                this.input_dict[n+k]=n+k;
            this.output_dict[n+this.samid]=n+this.samid; # the id of the sample
            this.output_dict[n+this.path_ids]=n+this.path_ids; # the id of path (we have n_sam* n_act)
            this.output_dict[n+this.out_logp]=n+this.out_logp; # the probability of that path
            this.output_dict[n+this.out_dlogp]=n+this.out_dlogp; # the detached log probability of that path



    def sample(this,logits):
        pass;
    # def take_action_impl(this,workspace:neko_workspace,environment:neko_environment):
    #     pass;
    #
    # # the world shall not backpropagate from routed items.
    # def take_action(this,workspace:neko_workspace,environment:neko_environment):
    #     with torch.no_grad():
    #         return this.take_action_impl(workspace,environment);

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.actions = this.register_input(this.INPUT_actions, iocvt_dict);
        this.in_logp = this.register_input(this.INPUT_raw_log_prob, iocvt_dict);
        this.out_logp=neko_get_arg(this.OUTPUT_weightath_log_prob,iocvt_dict);
        this.out_dlogp = neko_get_arg(this.OUTPUT_weightath_detached_log_prob,iocvt_dict);

        this.path_ids=this.register(this.OUTPUT_weightath_ids,iocvt_dict,this.output_dict);
        this.samid=this.register(this.OUTPUT_sample_id,iocvt_dict,this.output_dict);
    def make_path(this,workspace:neko_workspace,environment:neko_environment):
        na = workspace.get(this.actions);
        log_probs = workspace.get(this.in_logp);

        lk = {};
        path_id = 0;
        for item_id in range(len(na)):
            for eid in na[item_id]:
                target_anchor_name = this.names[eid];
                if (target_anchor_name not in lk):
                    lk[target_anchor_name] = 0;
                lk[target_anchor_name] += 1;
                tarname = this.tarname(this.samid, target_anchor_name);
                workspace.append_add(tarname, item_id);

                pathname = this.tarname(this.path_ids, target_anchor_name);
                workspace.append_add(pathname,
                                     path_id);  # one sample can be processed in more than one paths. hence this.s

                path_logprob_name = this.tarname(this.out_logp, target_anchor_name);
                workspace.append_add(path_logprob_name, log_probs[item_id, eid]);

                path_detached_logprob_name = this.tarname(this.out_dlogp, target_anchor_name);
                workspace.append_add(path_detached_logprob_name, log_probs[item_id, eid].detach());
                path_id += 1;
        workspace.logdict["routing"] = {"stat": lk, "routing_table": str(na)};
    def shipstuff(this,workspace:neko_workspace,environment:neko_environment):
        na = workspace.get(this.actions);
        for k in this.routing_subjects:
            if(k not in workspace.inter_dict):
                continue; # if target missing, don't ship it. # only used it in release mod.
                #  needed only if BOTH Athena subsystem and diff visualize are necessary.
            tart = workspace.inter_dict[k];
            for item_id in range(len(na)):
                for eid in na[item_id]:
                    target_anchor_name = this.names[eid];
                    tarname = this.tarname(k, target_anchor_name);
                    workspace.append_add(tarname, tart[item_id]);
            # cleanup after routing.
            for pfx in this.names:
                tarname = this.prefix + pfx + k.replace(this.prefix, "");
                if (tarname in workspace.inter_dict):
                    workspace.stackiftensor(tarname);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        this.make_path(workspace,environment);
        this.shipstuff(workspace,environment);
        return workspace, environment;
def get_neko_single_step_name_based_routing_agent_static(actions_name,raw_log_prob_name,
path_detached_log_prob_name,path_ids_name,path_log_prob_name,sample_id_name,
names,prefix,routing_subjects):
    engine = neko_single_step_name_based_routing_agent_static;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_actions: actions_name, engine.INPUT_raw_log_prob: raw_log_prob_name, engine.OUTPUT_weightath_detached_log_prob: path_detached_log_prob_name, engine.OUTPUT_weightath_ids: path_ids_name, engine.OUTPUT_weightath_log_prob: path_log_prob_name, engine.OUTPUT_sample_id: sample_id_name}, engine.PARAM_names: names, engine.PARAM_prefix: prefix, engine.PARAM_routing_subjects: routing_subjects, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_single_step_name_based_routing_agent_static.get_default_configuration_scripts());