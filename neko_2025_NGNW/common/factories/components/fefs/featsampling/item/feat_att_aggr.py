import torch

from neko_2025_NGNW.common.modules.quick_and_ugly_sp import neko_quick_and_dirty_sp
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN,P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.libimfe.agents.map2vec import neko_map_tower_to_vec_aaggr_agent_builder,neko_simple_map_tower_to_vec_aaggr_agent_builder
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk2 import spatial_attention_NG_mk2
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se_NG
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory, global_mod_cfg, \
    virtual_part_factory
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_agent

class neko_f2v_aagr_mod_factory(virtual_mod_factory):
    def arm_module(this, mod_prfx_dict, modcfg, bogocfg, params=None):
        prfx = mod_prfx_dict[this.MOD_PRFX_local];
        modcfg = this.config_saveable(
            modcfg, neko_add_embint_se_NG, {
                neko_add_embint_se_NG.PARAM_emb_w: 16,
                neko_add_embint_se_NG.PARAM_emb_h: 16,
                neko_add_embint_se_NG.PARAM_emb_ch: this.gmodcfg.num_se_channels,
            }, MN.SE(prfx)
        );
        modcfg = this.config_saveable(
            modcfg,
            spatial_attention_NG_mk2,
            {spatial_attention_NG_mk2.PARAM_ifc: this.gmodcfg.num_se_channels + this.gmodcfg.fe_ochs[1],
             spatial_attention_NG_mk2.PARAM_nparts: this.gmodcfg.nparts,
             }, MN.ATTN(prfx));
        return modcfg, bogocfg;


class neko_f2v_aggr_agt_factory(virtual_agt_factory):
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        dataprfx=neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        modprfx =neko_get_arg(this.MOD_PRFX_local, mod_prfx_dict);
        agtcfg = awa.append_agent_to_cfg(agtcfg, AN.MKPROTO(agt_prfx), neko_map_tower_to_vec_aaggr_agent_builder.get_agtcfg(
            VN.I_FEAT_MAP(dataprfx), VN.FEAT_MAP_LST(dataprfx),
            VN.GLOBAL_ATT_MAP(dataprfx), VN.I_FEAT_VEC(dataprfx),
            MN.SE(modprfx), MN.ATTN(modprfx), "NEP_skipped_NEP"));
        return agtcfg;

class neko_f2v_aggr_part_factory(virtual_part_factory):
    def setup_factories(this):
        super().setup_factories();
        this.f2v_modf = neko_f2v_aagr_mod_factory(this.gmodcfg);
        this.f2v_agtf = neko_f2v_aggr_agt_factory();
