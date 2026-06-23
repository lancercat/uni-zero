from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,global_mod_cfg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN

from osocrNG.modular_agents_ocrNG.pred_subs.poswise_token_translator import poswise_translate_agent_mk2


# maybe we symbollink things to different endpoints in task? to keeps api taking less parameters---
# wait, maybe not, just do what you have to do-- if you which you can call factory with the same prefix---
# no point for abstraction at this level

class  single_char_translator_factory(virtual_agt_factory):
    DATA_PRFX_meta = "meta";
    # although we don't want end users to mess with the data out of the prediciton head, it does not hurt as this is internal api
    # def arm_flatten_core(this, lacfg,feat_data_prefix,seq_prefix,agt_prfx):
    #     lacfg=awa.append_agent_to_cfg(lacfg,AN.flatten_agent(agt_prfx),
    #                         neko_single_roi_flatten_agent.get_agtcfg(
    #                                       VN.DENSE_CLS_PRED_LOGIT(feat_data_prefix),
    #                                       VN.FLATTEN_ROI_LOGIT_SEQ(seq_prefix),
    #                                       VN.FLATTEN_MAP(seq_prefix))
    #     );
    #     return lacfg;
    def arm_translate_agt(this,lacfg,logit_prefix,meta_data_prefix,agt_prfx):
        lacfg=awa.append_agent_to_cfg(lacfg, AN.TRANS(agt_prfx), poswise_translate_agent_mk2.get_agtcfg(
           VN.DENSE_CLS_PRED_LOGIT(logit_prefix),VN.TDICT(meta_data_prefix),
            VN.PRED_TEXT(logit_prefix),VN.PRED_TOK(logit_prefix)));
        return lacfg;

    def arm_translate_core(this, lacfg, feat_data_prefix,seq_prefix,meta_data_prefix,agt_prfx):
        # lacfg = this.arm_flatten_core(lacfg, feat_data_prefix, seq_prefix,agt_prfx);
        lacfg = this.arm_translate_agt(lacfg, seq_prefix, meta_data_prefix,agt_prfx);
        return lacfg;
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        meta_prfx=neko_get_arg(this.DATA_PRFX_meta,data_prfx_dict);
        head_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);
        return this.arm_translate_core(agtcfg,head_prfx,head_prfx,meta_prfx,agt_prfx);
