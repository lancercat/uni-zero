

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.modular_agents_ocrNG.losses.lenpred_loss import get_per_inst_len_loss_agent_mk2
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.averging import get_avging_loss_agent_mk2

# since we are calling this over different parts in a net, we put them together.


def get_lenpred_loss( len_pred_name,tensor_gt_len_name,per_inst_len_loss_name,batch_len_loss_name,
                      loss_mod_name):
    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": [ "len_loss_per_inst","len_loss_per_inst_aggr"],
            "len_loss_per_inst": get_per_inst_len_loss_agent_mk2(tensor_gt_len_name, len_pred_name,
                                               per_inst_len_loss_name,
                                               loss_mod_name),
            "len_loss_per_inst_aggr": get_avging_loss_agent_mk2(per_inst_len_loss_name,batch_len_loss_name)
        }
    }
def get_perinst_lenpred_loss( len_pred_name,tensor_gt_len_name,per_inst_len_loss_name,batch_len_loss_name,
                      loss_mod_name):
    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": [ "len_loss_per_inst"],
            "len_loss_per_inst": get_per_inst_len_loss_agent_mk2(tensor_gt_len_name, len_pred_name,
                                               per_inst_len_loss_name,
                                               loss_mod_name),
        }
    }
