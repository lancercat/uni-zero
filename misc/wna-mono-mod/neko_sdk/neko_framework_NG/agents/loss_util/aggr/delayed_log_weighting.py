import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import logweighting_loss_agent_mk2_detach_weight
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class logweighting_loss_agent_mk2_detach_weight_alter_delayed(logweighting_loss_agent_mk2_detach_weight):
    PARAM_disable_till_eid="disable_till_eid";
    PARAM_disable_till_bid="disable_till_bid";
    def set_etc(this,param):
        super().set_etc(param);
        this.disable_till_eid=neko_get_arg(this.PARAM_disable_till_eid,param);
        this.disable_till_bid = neko_get_arg(this.PARAM_disable_till_bid, param);

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())  + this.base_weight;
        if (workspace.epoch_idx < this.disable_till_eid):
            weights = torch.ones_like(weights);
        elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
            weights = torch.ones_like(weights);
        return weights;

def get_logweighting_loss_agent_mk2_detach_weight_alter_delayed(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight,disable_till_eid,disable_till_bid):
    engine = logweighting_loss_agent_mk2_detach_weight_alter_delayed;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid,
        "modcvt_dict": {}}}

