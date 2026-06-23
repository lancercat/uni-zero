from neko_2024_NGNW.common.agent_pack.branches_mk3.branch_mk3 import neko_branch_mk3_single_head

from neko_sdk.neko_framework_NG.agents.loss_util.aggr.log_weighting import get_logweighting_loss_agent_mk2_detach_weight_list
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.ohem_delayed_log_weighting import \
    (get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01,
     get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01E,
     get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01m200)




class neko_branch_mk3_single_head_ohem01(neko_branch_mk3_single_head):
    def weight_loss(this, src, dst, weight, base_weight=0.1):
        return get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01(weight, src, dst, base_weight,
                                                                                  this.AGT_W_delay_e,
                                                                                  this.AGT_W_delay_b);

class neko_branch_mk3_single_head_ohem01E(neko_branch_mk3_single_head):
    def weight_loss(this, src, dst, weight, base_weight=0.1):
        return get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01E(weight, src, dst, base_weight,
                                                                                  this.AGT_W_delay_e,
                                                                                  this.AGT_W_delay_b);

class neko_branch_mk3_single_head_ohem01m200(neko_branch_mk3_single_head):
    def weight_loss(this, src, dst, weight, base_weight=0.1):
        return get_logweighting_loss_agent_mk2_detach_weight_alter_delayed_ohem01m200(weight, src, dst, base_weight,
                                                                                  this.AGT_W_delay_e,
                                                                                  this.AGT_W_delay_b);