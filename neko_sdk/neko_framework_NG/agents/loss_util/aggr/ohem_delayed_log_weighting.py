import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import logweighting_loss_agent_mk2_detach_weight
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.delayed_log_weighting import logweighting_loss_agent_mk2_detach_weight_alter_delayed


class logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01(logweighting_loss_agent_mk2_detach_weight_alter_delayed):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())  + this.base_weight;
        if (workspace.epoch_idx < this.disable_till_eid):
            weights = torch.ones_like(weights);
        elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
            weights = torch.ones_like(weights);
        l = workspace.get(this.item_loss);
        cnt = int(len(l) * this.too_hard)
        if(cnt>0):
            weights[torch.topk(l,cnt)[1]]=0; # drop the hardest 10% samples (maybe anno error or maybe too much aug)
        return weights;
class logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01_eff(logweighting_loss_agent_mk2_detach_weight_alter_delayed):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())  + this.base_weight;
            if (workspace.epoch_idx < this.disable_till_eid):
                weights = torch.ones_like(weights);
            elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
                weights = torch.ones_like(weights);
            l = workspace.get(this.item_loss);
            el = l * weights;
        cnt = int(len(l) * this.too_hard)
        if (cnt > 0):
            weights[torch.topk(l, cnt)[1]] = 0;  # drop the hardest 10% samples (maybe anno error or maybe too much aug)
        cnt = int(len(el) * this.too_hard)
        if(cnt>0):
            weights[torch.topk(el,cnt)[1]]=0; # drop the hardest 10% samples ACTUALL ROUTED to the agent. (maybe anno error or maybe too much aug)

        return weights;
class logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01m200(logweighting_loss_agent_mk2_detach_weight_alter_delayed):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;
        this.too_hard_stat = 0.1;
        this.momentum=torch.zeros([1],dtype=torch.float,device="cuda",requires_grad=False);
        this.keep_cnt=200; # keep 200 recent samples for loss computation


    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            l = workspace.get(this.item_loss);
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach()) + this.base_weight;
            if (workspace.epoch_idx < this.disable_till_eid):
                weights = torch.ones_like(weights);
            elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
                weights = torch.ones_like(weights);
            # normal ohem
            cnt = int(len(l) * this.too_hard)
            if (cnt > 0):
                weights[
                    torch.topk(l, cnt)[1]] = 0;  # drop the hardest 10% samples (maybe anno error or maybe too much aug)
            # surge control based on statistic loss
            this.momentum=torch.cat([l,this.momentum])[-this.keep_cnt:];
            kth=int((1-this.too_hard_stat)*len(this.momentum));
            hthres=this.momentum.kthvalue(kth)[0];
            weights[l>hthres]=0;
        return weights;
def get_logweighting_loss_agent_mk2_detach_weight(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight):
    engine = logweighting_loss_agent_mk2_detach_weight;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight,
        "modcvt_dict": {}}}
def get_logweighting_loss_agent_mk2_detach_weight_alter_delayed(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight,disable_till_eid,disable_till_bid):
    engine = logweighting_loss_agent_mk2_detach_weight_alter_delayed;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid,
        "modcvt_dict": {}}}
def get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight,disable_till_eid,disable_till_bid):
    engine = logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid,
        "modcvt_dict": {}}}

def get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01E(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight,disable_till_eid,disable_till_bid):
    engine = logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01_eff;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid,
        "modcvt_dict": {}}}
def get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01m200(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight,disable_till_eid,disable_till_bid):
    engine = logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem_01m200;
    return {"agent": engine, "params": {
        "iocvt_dict": {
            engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name},
        engine.PARAM_base_weight: base_weight, engine.PARAM_disable_till_bid: disable_till_bid, engine.PARAM_disable_till_eid: disable_till_eid,
        "modcvt_dict": {}}}
if __name__ == '__main__':
    print(logweighting_loss_agent_mk2_detach_weight_alter_delayed.get_default_configuration_scripts());