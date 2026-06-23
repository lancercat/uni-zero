from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg


from neko_2025_NGNW.common.factories.open_set_classification import classification_factory_mk2_mod, \
    classification_factory_mk2_agt



from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN



from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory,global_mod_cfg
from neko_2025_NGNW.common.factories.components.heads.losses_and_rewards.single_char_cls_loss import single_char_cls_mod_factory,single_char_cls_agt_factory
from neko_2025_NGNW.common.factories.components.heads.gtmker.basic_osocr_gtmker import neko_basic_gt_mker_agt_factory
from neko_2025_NGNW.common.factories.components.heads.translator.single_char import single_char_translator_factory

from osocrNG.modular_agents_ocrNG.aggrate_agents.char_flatten import neko_single_roi_flatten_agent
from neko_2025_NGNW.common.factories.typical_parts.heads.virtual_head import neko_object320_virtual_head_factory

# this thing will provide many builds--- bcs we have different needs
# this thing will not protofy the meta tho. you use the as_proto factory outside to sort that out.
# we have unified meta and data as much as we can, however in 320 main ocr task will still differentiate their standings---
# so in the future we will need to make this better and simpler.


# since mk2 training decoding is delegated by an external baseclass.
# class boundary is managed per head. you can do that per meta in later revision.
class neko_object320_osic_head_factory(neko_object320_virtual_head_factory):
    def set_classification_factories(this):
        this.classification_modf= classification_factory_mk2_mod(this.gmodcfg, {});
        this.classification_agtf = classification_factory_mk2_agt();
    def set_gt_mker_factories(this):
        this.gtmker=neko_basic_gt_mker_agt_factory(); # there will be len_gt anyways, just don't use it :)
    def set_loss_factories(this):
        this.loss_modf=single_char_cls_mod_factory(this.gmodcfg,{});
        this.loss_agtf=single_char_cls_agt_factory();
    def set_translation_factories(this):
        this.trans_agtf=single_char_translator_factory();


    def __init__(this,platformcfg:neko_platform_cfg,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(platformcfg,gmodcfg,external_factory_dict);
        this.set_classification_factories();
        this.set_translation_factories();
        this.set_gt_mker_factories();
        this.set_loss_factories();

    def arm_head_mods(this, modcfg, bogocfg, endpoint_mod_prfx,tcfg):
        this.classification_modf.arm_module({this.classification_modf.MOD_PRFX_local: endpoint_mod_prfx }, modcfg,bogocfg);
        this.loss_modf.arm_module({this.loss_modf.MOD_PRFX_local:endpoint_mod_prfx},modcfg,bogocfg);
        return modcfg, bogocfg;
    def arm_pred_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix):

        this.classification_agtf.arm_agt_core({
            this.classification_agtf.MOD_PRFX_local:endpoint_mod_prfx,
        }, {
            this.classification_agtf.DATA_PRFX_meta: meta_data_prefix,
            this.classification_agtf.DATA_PRFX_feat: feat_data_prfx,
            this.classification_agtf.DATA_PRFX_local: endpoint_data_prefix
        }, acfg, AN.DCLS(endpoint_data_prefix), {});

        this.trans_agtf.arm_agt_core({},{
            this.trans_agtf.DATA_PRFX_meta:meta_data_prefix,
            this.trans_agtf.DATA_PRFX_local:endpoint_data_prefix
        },acfg,AN.TRANS(endpoint_data_prefix),{});
        return acfg;
    # may our maynot use data from pred agent,
    # but will always assume that pred agents have been called.
    # don't make label unless you have loss.
    def arm_training_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix,training_endpoint_data_prfx):
        acfg = awa.append_agent_to_cfg(acfg,P(training_endpoint_data_prfx,"link_pred_logit"),neko_symbol_link_alot_agent.get_agtcfg(
            [VN.DENSE_CLS_PRED_LOGIT(endpoint_data_prefix)],
            [VN.DENSE_CLS_PRED_LOGIT(training_endpoint_data_prfx)]));
        acfg = awa.append_agent_to_cfg(acfg, AN.flatten_agent(P(training_endpoint_data_prfx,"flatten")),
                                neko_single_roi_flatten_agent.get_agtcfg(
                                               VN.DENSE_CLS_PRED_LOGIT(training_endpoint_data_prfx),
                                               VN.FLATTEN_ROI_LOGIT_SEQ(training_endpoint_data_prfx),
                                               VN.FLATTEN_MAP(training_endpoint_data_prfx))
             ); # oscls training loss depends on flattend gt to work.
        acfg = this.gtmker.arm_agt_core(
            {},
            {this.gtmker.DATA_PRFX_meta:meta_data_prefix,this.gtmker.DATA_PRFX_feat:feat_data_prfx,this.gtmker.DATA_PRFX_local:training_endpoint_data_prfx},
            acfg,training_endpoint_data_prfx);
        acfg= this.loss_agtf.arm_agt_core({this.loss_agtf.MOD_PRFX_local:endpoint_mod_prfx},
                                          {this.loss_agtf.DATA_PRFX_local:training_endpoint_data_prfx},acfg,training_endpoint_data_prfx);
        return acfg;

