from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg


from neko_2025_NGNW.common.factories.open_set_classification import classification_factory_mk2_mod, \
    classification_factory_mk2_agt

from neko_sdk.neko_framework_NG.agents.loss_util.aggr.averging import avging_loss_agent_mk2


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN


from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory,global_mod_cfg
from neko_2025_NGNW.common.factories.components.heads.pred.len_time_align_seq import position_wise_with_length_pred_mod_factory,position_wise_with_length_pred_agt_factory,position_wise_with_length_pred_training_agt_factory
from neko_2025_NGNW.common.object_32x_presets.templates.task_processing_profile import neko_task_processing,neko_seq_task_param

# this thing will provide many builds--- bcs we have different needs
# this thing will not protofy the meta tho. you use the as_proto factory outside to sort that out.
# we have unified meta and data as much as we can, however in 320 main ocr task will still differentiate their standings---
# so in the future we will need to make this better and simpler.
from neko_2025_NGNW.common.factories.typical_parts.heads.virtual_head import neko_object320_virtual_head_factory


# since mk2 training decoding is delegated by an external baseclass.
# class boundary is managed per head. you can do that per meta in later revision.
class neko_object320_oslpos_head_factory(neko_object320_virtual_head_factory):

    def __init__(this,platformcfg:neko_platform_cfg,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(platformcfg,gmodcfg,external_factory_dict);
        this.lcam_modf=position_wise_with_length_pred_mod_factory(gmodcfg,{});
        this.pred_lcam_agtf=position_wise_with_length_pred_agt_factory();
        this.train_lcam_agtf=position_wise_with_length_pred_training_agt_factory();
        this.classification_modf= classification_factory_mk2_mod(this.gmodcfg, {});
        this.classification_agtf = classification_factory_mk2_agt();

    def arm_head_mods(this, modcfg, bogocfg, endpoint_mod_prfx,tcfg):
        this.classification_modf.arm_module({this.classification_modf.MOD_PRFX_local: endpoint_mod_prfx }, modcfg,bogocfg);
        maxT=tcfg[neko_task_processing.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_MAXT];
        this.lcam_modf.arm_module({this.lcam_modf.MOD_PRFX_local: endpoint_mod_prfx }, modcfg,bogocfg,{this.lcam_modf.PARAM_maxT:maxT});
        return modcfg, bogocfg;
    def arm_pred_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix):

        this.classification_agtf.arm_agt_core({
            this.classification_agtf.MOD_PRFX_local:endpoint_mod_prfx,
        }, {
            this.classification_agtf.DATA_PRFX_meta: meta_data_prefix,
            this.classification_agtf.DATA_PRFX_feat: feat_data_prfx,
            this.classification_agtf.DATA_PRFX_local: endpoint_data_prefix
        }, acfg, AN.DCLS(endpoint_data_prefix), {});

        this.pred_lcam_agtf.arm_agt_core({
            this.pred_lcam_agtf.MOD_PRFX_local:endpoint_mod_prfx,
        }, {
            this.pred_lcam_agtf.DATA_PRFX_meta: meta_data_prefix,
            this.pred_lcam_agtf.DATA_PRFX_roiseq_feat: feat_data_prfx,
            this.pred_lcam_agtf.DATA_PRFX_local: endpoint_data_prefix
        }, acfg, AN.DCLS(endpoint_data_prefix), {});
        return acfg;
    # may our maynot use data from pred agent,
    # but will always assume that pred agents have been called.
    # don't make label unless you have loss.
    def arm_training_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix,training_endpoint_data_prfx):
        acfg = awa.append_agent_to_cfg(acfg,P(training_endpoint_data_prfx,"link_pred_logit"),neko_symbol_link_alot_agent.get_agtcfg(
            [VN.DENSE_CLS_PRED_LOGIT(endpoint_data_prefix)],
            [VN.DENSE_CLS_PRED_LOGIT(training_endpoint_data_prfx)]));
        acfg=this.train_lcam_agtf.arm_agt_core(
            {this.train_lcam_agtf.MOD_PRFX_local:endpoint_mod_prfx},{
            this.train_lcam_agtf.DATA_PRFX_meta: meta_data_prefix,
            this.train_lcam_agtf.DATA_PRFX_pred_endpoint: endpoint_data_prefix,
            this.train_lcam_agtf.DATA_PRFX_roiseq_feat: feat_data_prfx,
            this.train_lcam_agtf.DATA_PRFX_local: training_endpoint_data_prfx,
            this.train_lcam_agtf.DATA_PRFX_gt:feat_data_prfx,
        },acfg,P(training_endpoint_data_prfx,"loss_and_penalty"));
        return acfg;
