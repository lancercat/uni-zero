import torch.nn
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf

# We will make parameter free module agents, to reduce coding complexity.
# Collects penalties from all possible decisions.
# Decisions that did not happen has infinite_penalty.

class neko_collect_penalty(neko_module_wrapping_agent):
    INPUT_penalty_name = "penalty";
    INPUT_item_id = "item_id";
    PARAM_anchors="anchors";
    INPUT_router_logits = "router_logits";
    OUTPUT_penaltyspace = "penaltyspace";
    OUTPUT_penaltymask = "penaltymask";

    OUTPUT_weight = "router_logits";
    PARAM_inf_penalty="inf_penalty";
    def set_etc(this,param):
        this.anchors=neko_get_arg(this.PARAM_anchors,param);
        this.inf_penalty=neko_get_arg(this.PARAM_inf_penalty,param,9999);

        for a in this.anchors:
            this.input_dict[a+this.INPUT_item_id]=a+this.item_id;
            this.input_dict[a + this.INPUT_penalty_name] = a + this.penalty;
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.router_logits = this.register_input(this.INPUT_router_logits, iocvt_dict);
        this.item_id=neko_get_arg(this.INPUT_item_id,iocvt_dict);
        this.penalty= neko_get_arg(this.INPUT_penalty_name,iocvt_dict);
        this.penaltyspace = this.register_output(this.OUTPUT_penaltyspace, iocvt_dict);
        this.penaltymask = this.register_output(this.OUTPUT_penaltymask, iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        router_logits=workspace.get(this.router_logits);
        with torch.no_grad():
            penaltyspace=torch.zeros_like(router_logits)+this.inf_penalty;
            penaltymask=torch.zeros_like(router_logits)+1;

            for aid in range(len(this.anchors)):
                if(this.anchors[aid]+this.item_id not in workspace.inter_dict):
                    penaltymask[:, aid][item_id]=0;
                    continue;
                item_id=workspace.get(this.anchors[aid]+this.item_id);
                penalty=workspace.get(this.anchors[aid]+this.penalty);
                penaltyspace[:,aid][item_id]=penalty;

        workspace.add(this.penaltyspace,penaltyspace);
        workspace.add(this.penaltymask,penaltymask);

    @classmethod
    def get_agtcfg(cls,
                   item_id, penalty_name, router_logits,
                   penaltymask, penaltyspace, weight,
                   anchors, inf_penalty
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_item_id: item_id, cls.INPUT_penalty_name: penalty_name,
                           cls.INPUT_router_logits: router_logits, cls.OUTPUT_penaltymask: penaltymask,
                           cls.OUTPUT_penaltyspace: penaltyspace, cls.OUTPUT_weight: weight},
            cls.PARAM_anchors: anchors, cls.PARAM_inf_penalty: inf_penalty, "modcvt_dict": {}}}


class neko_supervised_on_min_penalty(neko_module_wrapping_agent):
    INPUT_penalty_name = "penalty";
    INPUT_item_id="item_id";
    INPUT_router_logits = "router_logits";

    OUTPUT_loss = "selection_loss";
    OUTPUT_actual_best="actual_best";
    PARAM_anchors="anchors";
    PARAM_disable_till_eid="disable_till_eid";
    PARAM_disable_till_bid="disable_till_bid";
    def set_etc(this,param):
        this.anchors=neko_get_arg(this.PARAM_anchors,param);
        this.disable_till_eid=neko_get_arg(this.PARAM_disable_till_eid,param,0);
        this.disable_till_bid = neko_get_arg(this.PARAM_disable_till_bid, param, 0);

        for a in this.anchors:
            this.input_dict[a+this.INPUT_item_id]=a+this.item_id;
            this.input_dict[a + this.INPUT_penalty_name] = a + this.penalty;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.router_logits = this.register_input(this.INPUT_router_logits, iocvt_dict);
        this.item_id=neko_get_arg(this.INPUT_item_id,iocvt_dict);
        this.penalty= neko_get_arg(this.INPUT_penalty_name,iocvt_dict);

        this.loss = this.register_output(this.OUTPUT_loss, iocvt_dict);
        this.actual_best=this.register_output(this.OUTPUT_actual_best,iocvt_dict);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if (workspace.epoch_idx < this.disable_till_eid):
            return workspace, environment;
        if (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
            return workspace, environment;
        router_logits=workspace.get(this.router_logits);
        with torch.no_grad():
            penaltyspace=torch.zeros_like(router_logits)+9999;
            for aid in range(len(this.anchors)):
                if(this.anchors[aid]+this.item_id not in workspace.inter_dict):
                    continue;
                item_id=workspace.get(this.anchors[aid]+this.item_id);
                penalty=workspace.get(this.anchors[aid]+this.penalty);
                penaltyspace[:,aid][item_id]=penalty;
            label=penaltyspace.argmin(-1);
            weight=(penaltyspace.min(-1)[0]<9999).float();
            weight=weight/weight.sum()+0.000001
        loss=trnf.cross_entropy(router_logits,label,reduction="none")*weight;
        loss=loss.sum();
        workspace.objdict[this.loss]=loss;
        workspace.logdict[this.loss]=loss.item();

        workspace.logdict[this.actual_best]=[this.anchors[i] for i in label.detach().cpu()[:3]]
        return workspace, environment;


def get_neko_supervised_on_min_penalty(item_id_name,penalty_name,router_logits_name,
actual_best_name,loss_name,
anchors,disable_till_bid,disable_till_eid):
    engine = neko_supervised_on_min_penalty;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_item_id: item_id_name, engine.INPUT_penalty_name: penalty_name, engine.INPUT_router_logits: router_logits_name, engine.OUTPUT_actual_best: actual_best_name, engine.OUTPUT_loss: loss_name}, engine.PARAM_anchors: anchors, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_collect_penalty.print_default_setup_scripts()
