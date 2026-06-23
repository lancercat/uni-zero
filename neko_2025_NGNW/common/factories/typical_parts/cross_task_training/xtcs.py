from neko_2025_NGNW.common.agents.training_and_testing.cross_task_cotraining.xtcs import neko_simple_xtcs_agent
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg


from neko_2025_NGNW.common.factories.open_set_classification import classification_factory_mk2_mod, \
    classification_factory_mk2_agt

from neko_sdk.neko_framework_NG.agents.loss_util.aggr.averging import avging_loss_agent_mk2


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_2025_NGNW.common.object_32x_presets.templates.task_processing_profile import neko_task_processing,neko_xcts_task_param


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awa_nograd

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory,global_mod_cfg
from neko_2025_NGNW.common.agents.training_and_testing.cross_task_cotraining.xtcs import neko_simple_xtcs_agent
class neko_cross_task_cos_sim_factory(virtual_part_factory):
    def arm_training_agent(this,acfg,trainingcfg):
        taskprfx=trainingcfg[neko_task_processing.NAME];
        param=trainingcfg[neko_task_processing.TYPE_PARAM];
        frozen_item_feat_list=param[neko_xcts_task_param.TASK_XCTS_GT_ITEM_FEAT_EPS];
        frozen_item_utf_list = param[neko_xcts_task_param.TASK_XCTS_GT_ITEM_UTF_EPS];
        item_feat_list = param[neko_xcts_task_param.TASK_XCTS_ITEM_FEAT_EPS];
        item_utf_list = param[neko_xcts_task_param.TASK_XCTS_ITEM_UTF_EPS];
        seq_feat_list=param[neko_xcts_task_param.TASK_XCTS_SEQ_FEAT_EPS];
        seq_utf_list = param[neko_xcts_task_param.TASK_XCTS_SEQ_UTF_EPS];
        cos_sim_loss=VN.COSSIM(taskprfx);
        TDICT=param[neko_xcts_task_param.TASK_XCTS_GTDICTS];

        acfg = awa.append_agent_to_cfg(acfg,P(taskprfx,"cross_task_class_sim"),neko_simple_xtcs_agent.get_agtcfg(frozen_item_feat_list, frozen_item_utf_list, item_feat_list, item_utf_list, seq_feat_list,
                   seq_utf_list,
                   cos_sim_loss,
                   TDICT));
        return acfg;


    pass;
