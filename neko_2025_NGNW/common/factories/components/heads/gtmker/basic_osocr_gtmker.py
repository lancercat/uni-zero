from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awn
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_mod_factory,virtual_agt_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from neko_sdk.OSR.classification.agents.neko_missing_token_to_unk import neko_missing_to_unk

from osocrNG.modular_agents_ocrNG.ocr_label_agents_g2.neko_ccd_label_making_agent import \
     neko_ccd_label_making_agent_mk2s
class neko_basic_gt_mker_agt_factory(virtual_agt_factory):
    DATA_PRFX_meta= "data_prfx_meta";
    DATA_PRFX_feat="data_prfx_feat";
    def get_mkgt(this, head_training_prfx, feat_prfx,meta_prfx):
        lacfg=awn.empty([VN.GT_TOK_UTF(feat_prfx)]); #
        # tokenizer is defaulted to just steal tokenized stuff from upstream.
        lacfg = awa.append_agent_to_cfg(lacfg, "tokenize",
                                       neko_symbol_link_alot_agent.get_agtcfg([VN.GT_TOK_UTF(feat_prfx)],
                                                                                   [VN.GT_TOK_UTF(head_training_prfx)]
                                                                                   ));
        lacfg=awn.append_agent_to_cfg(lacfg,"decorate_unk",neko_missing_to_unk.get_agtcfg(
            VN.GT_TOK_UTF(head_training_prfx),VN.TDICT(meta_prfx),
            VN.GT_TOK_UTF_WUNK(head_training_prfx)));
        lacfg=awn.append_agent_to_cfg(lacfg,"get_tensor_gt",
            neko_ccd_label_making_agent_mk2s.get_agtcfg(
                VN.PROTO(meta_prfx),VN.GT_TOK_UTF_WUNK(head_training_prfx),VN.TDICT(meta_prfx),
                VN.FLATTEN_ALIGNED_TLABEL(head_training_prfx),VN.WORD_LEN_GT_VALUE(head_training_prfx)
        ));# t
        # it does not use plabel--- reduction is done in head.
        return lacfg
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        head_training_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);
        meta_prfx = neko_get_arg(this.DATA_PRFX_meta, data_prfx_dict);
        feat_prfx = neko_get_arg(this.DATA_PRFX_feat, data_prfx_dict);
        return awa.append_agent_to_cfg(agtcfg,AN.MKGT(agt_prfx),this.get_mkgt(head_training_prfx,feat_prfx,meta_prfx));
