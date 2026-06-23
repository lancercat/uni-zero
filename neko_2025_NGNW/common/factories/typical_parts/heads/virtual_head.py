from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_2025_NGNW.common.factories.open_set_classification import classification_factory_mk2_mod, \
    classification_factory_mk2_agt

from neko_sdk.neko_framework_NG.agents.loss_util.aggr.averging import avging_loss_agent_mk2

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awa_nograd

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory, global_mod_cfg

class neko_object320_virtual_head_factory(virtual_part_factory):

    def arm_head_mods(this, modcfg, bogocfg, endpoint_mod_prfx, tcfg):
        return modcfg, bogocfg;

    def arm_pred_agent(this, acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx, endpoint_data_prefix):
        return acfg;

    # may our maynot use data from pred agent,
    # but will always assume that pred agents have been called.
    # don't make label unless you have loss.
    def arm_training_agent(this, acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx, endpoint_data_prefix,
                           training_endpoint_data_prfx):
        return acfg;

    # if you don't want gradient from this--- for whatever reason. For example if you are using seq2seq this can help you save some time and vram in reward computation
    # otherwise in case of ctc and char level training you may want to mux the training and testing prediction to speed up.
    def arm_pred_agent_nograd(this, acfg, meta_data_prefix, meta_mod_prfx, feat_data_prfx, endpoint_data_prefix):
        lacfg = awa_nograd.empty();
        lacfg = this.arm_pred_agent(lacfg, meta_data_prefix, meta_mod_prfx, feat_data_prfx, endpoint_data_prefix)
        return awa.append_agent_to_cfg(acfg, AN.PRED(endpoint_data_prefix), lacfg);

    # bcs wether you want gradient from the prediction branch depends what kind of implementation you have--- so they need to be constructed in a bundle
    # and if you don't use prediction front for gradient during trianing, override to arm the nograd branch, or just don't arm if you don't even need the translation for penalty computation or what soever...
    def arm_training_and_pred_agents(this, acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx,
                                     pred_endpoint_data_prefix, training_endpoint_data_prfx):
        acfg = this.arm_pred_agent(acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx,
                                   pred_endpoint_data_prefix);
        acfg = this.arm_training_agent(acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx,
                                       pred_endpoint_data_prefix, training_endpoint_data_prfx);
        lacfg = awa.empty([VN.LOSS_PER_INST(training_endpoint_data_prfx)]); # if the loss does not present - for semisupervised cotraining or some time the gt gen on complex synth data fails.
        lacfg=awa.append_agent_to_cfg(lacfg, "task_loss_total",
                                       avging_loss_agent_mk2.get_agtcfg(VN.LOSS_PER_INST(training_endpoint_data_prfx),
                                                                        VN.TASK_LOSS(training_endpoint_data_prfx)));
        awa.append_agent_to_cfg( acfg,P(pred_endpoint_data_prefix,"task_loss_total"),lacfg);
        return acfg;
