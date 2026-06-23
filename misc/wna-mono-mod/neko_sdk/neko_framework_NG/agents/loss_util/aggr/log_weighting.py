
import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.averging import avging_loss_agent_mk2

class logweighting_loss_agent_mk2(avging_loss_agent_mk2):
    INPUT_per_item_log_weight_name="input_log_weight_name";
    PARAM_base_weight="base_weight";

    def set_etc(this,params):
        this.base_weight=neko_get_arg(this.PARAM_base_weight,params,0.1);
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.item_log_weight=this.register_input(this.INPUT_per_item_log_weight_name,iocvt_dict);
        this.item_loss=this.register_input(this.INPUT_per_item_loss_name,iocvt_dict);
        this.lossname=this.register_output(this.OUTPUT_loss_name,iocvt_dict);
    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        weights=torch.exp(workspace.get(this.item_log_weight))+this.base_weight;
        return weights;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_loss =workspace.get(this.item_loss);
        weights=this.get_weights(workspace,environment);
        loss=torch.mean(weights*per_inst_loss);
        # will this actually discourage large traffic over an agent?
        # not if you detach your initial weights...
        workspace.add_loss(this.lossname,loss);
        workspace.logdict[this.lossname]=loss.item();
        return workspace, environment;
def get_logweighting_loss_agent_mk2(per_item_log_weight_name,per_item_loss_name,
loss_name,
base):
    engine = logweighting_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name}, engine.PARAM_base: base, "modcvt_dict": {}}}
class logweighting_loss_agent_mk2_detach_weight(logweighting_loss_agent_mk2):
    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        weights = torch.exp(workspace.get(this.item_log_weight).detach()) + this.base_weight;
        return weights;
def get_logweighting_loss_agent_mk2_detach_weight(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight):
    engine = logweighting_loss_agent_mk2_detach_weight;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name}, engine.PARAM_base_weight: base_weight, "modcvt_dict": {}}}

class logweighting_loss_agent_mk2_detach_weight_list(logweighting_loss_agent_mk2_detach_weight):
    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach()) + this.base_weight;
        return weights;
def get_logweighting_loss_agent_mk2_detach_weight_list(per_item_log_weight_name,per_item_loss_name,
loss_name,
base_weight):
    engine = logweighting_loss_agent_mk2_detach_weight_list;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_per_item_log_weight_name: per_item_log_weight_name, engine.INPUT_per_item_loss_name: per_item_loss_name, engine.OUTPUT_loss_name: loss_name}, engine.PARAM_base_weight: base_weight, "modcvt_dict": {}}}


class logweighting_loss_agent_mk2_detach_weight_gf_list(logweighting_loss_agent_mk2):
    PARAM_global_factor = "global_factor";

    def set_etc(this, params):
        super().set_etc(params);
        this.global_factor = neko_get_arg(this.PARAM_global_factor, params, 1);

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())*this.global_factor + this.base_weight;
        return weights;


def get_logweighting_loss_agent_mk2_detach_weight_gf_list(per_item_log_weight_name, per_item_loss_name,
                                                     loss_name,
                                                     base_weight,global_factor):
    engine = logweighting_loss_agent_mk2_detach_weight_gf_list;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_per_item_log_weight_name: per_item_log_weight_name,
                                                       engine.INPUT_per_item_loss_name: per_item_loss_name,
                                                       engine.OUTPUT_loss_name: loss_name},
                                        engine.PARAM_base_weight: base_weight,
                                        engine.PARAM_global_factor: global_factor,
                                        "modcvt_dict": {}}}
