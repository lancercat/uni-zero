import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import logweighting_loss_agent_mk2_detach_weight_list
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import logweighting_loss_agent_mk2

class logweighting_loss_agent_mk3_detach_weight_ohem_01(logweighting_loss_agent_mk2):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            weights = torch.exp(workspace.get(this.item_log_weight).detach())  + this.base_weight;
            l = workspace.get(this.item_loss);
            cnt = int(len(l) * this.too_hard)
            if(cnt>0):
                weights[torch.topk(l,cnt)[1]]=0; # drop the hardest 10% samples (maybe anno error or maybe too much aug)
            return weights;

    @classmethod
    def get_agtcfg(cls,
                   per_item_log_weight_name, per_item_loss_name,
                   loss_name,
                   base_weight
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_per_item_log_weight_name: per_item_log_weight_name,
                                                        cls.INPUT_per_item_loss_name: per_item_loss_name,
                                                        cls.OUTPUT_loss_name: loss_name},
                                         cls.PARAM_base_weight: base_weight, "modcvt_dict": {}}}


class logweighting_loss_agent_mk3_detach_weight_ohem_01_eff(logweighting_loss_agent_mk3_detach_weight_ohem_01):
    def set_etc(this, param):
        super().set_etc(param);
        this.too_hard=0.1;

    def get_weights(this, workspace: neko_workspace, environment: neko_environment):
        with torch.no_grad():
            l = workspace.get(this.item_loss);

            if(this.item_log_weight in workspace.inter_dict):
                weights = torch.exp(workspace.get(this.item_log_weight).detach()) + this.base_weight;
            else:
                weights= torch.zeros_like(l)+this.base_weight; # if the sample is not weighted, assume equal
            el = l * weights;
            cnt = int(len(l) * this.too_hard)
            if (cnt > 0):
                weights[torch.topk(l, cnt)[1]] = 0;  # drop the hardest 10% samples (maybe anno error or maybe too much aug)
            cnt = int(len(el) * this.too_hard)
            if (cnt > 0):
                weights[torch.topk(el, cnt)[1]] = 0;  # drop the hardest 10% samples ACTUALL ROUTED to the agent. (maybe anno error or maybe too much aug)

        return weights;
if __name__ == '__main__':
    logweighting_loss_agent_mk3_detach_weight_ohem_01_eff.print_default_setup_scripts()