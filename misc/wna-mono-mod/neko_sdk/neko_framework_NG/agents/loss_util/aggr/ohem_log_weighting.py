import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import logweighting_loss_agent_mk2_detach_weight_list
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class logweighting_loss_agent_mk2_detach_weight_ohem_01(logweighting_loss_agent_mk2_detach_weight_list):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())  + this.base_weight;
            l = workspace.get(this.item_loss);
            cnt = int(len(l) * this.too_hard)
            if(cnt>0):
                weights[torch.topk(l,cnt)[1]]=0; # drop the hardest 10% samples (maybe anno error or maybe too much aug)
            return weights;
class logweighting_loss_agent_mk2_detach_weight_ohem_01_eff(logweighting_loss_agent_mk2_detach_weight_list):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach()) + this.base_weight;
            l = workspace.get(this.item_loss);
            el = l * weights;
            cnt = int(len(l) * this.too_hard)
            if (cnt > 0):
                weights[torch.topk(l, cnt)[1]] = 0;  # drop the hardest 10% samples (maybe anno error or maybe too much aug)
            cnt = int(len(el) * this.too_hard)
            if (cnt > 0):
                weights[torch.topk(el, cnt)[1]] = 0;  # drop the hardest 10% samples ACTUALL ROUTED to the agent. (maybe anno error or maybe too much aug)

        return weights;


class logweighting_loss_agent_mk2_detach_weight(logweighting_loss_agent_mk2_detach_weight_list):
    def set_etc(this,param):
        pass;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())  + this.base_weight;
        return weights;
def get_logweighting_loss_agent_mk2_detach_weight_ohem01(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight):
    # using routing proababilty from  the first iteration--even if that means to itroduce bias on very early stage.
    engine = logweighting_loss_agent_mk2_detach_weight_ohem_01;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight,
        "modcvt_dict": {}}}
def get_logweighting_loss_agent_mk2_detach_weight_ohem01E(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight):
    # using routing proababilty from  the first iteration--even if that means to itroduce bias on very early stage.
    engine = logweighting_loss_agent_mk2_detach_weight_ohem_01_eff;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight,
        "modcvt_dict": {}}}