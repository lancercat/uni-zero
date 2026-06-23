import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN,P

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.libimfe.agents.map2vec import neko_simple_map_tower_to_vec_aaggr_agent_builder
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk2 import spatial_attention_NG_mk2
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory, global_mod_cfg, \
    virtual_part_factory
class neko_simple_f2v_aagr_mod_factory(virtual_mod_factory):

    def __init__(this, gmodcfg: global_mod_cfg, external_factory_dict=None):
        super().__init__(gmodcfg, external_factory_dict);

    def arm_module(this, mod_prfx_dict, modcfg, bogocfg, params=None):
        prfx = mod_prfx_dict[this.MOD_PRFX_local];
        modcfg = this.config_saveable(
            modcfg,
            spatial_attention_NG_mk2,
            {spatial_attention_NG_mk2.PARAM_ifc: this.gmodcfg.fe_ochs[1],
             spatial_attention_NG_mk2.PARAM_nparts: this.gmodcfg.nparts,
             }, MN.ATTN(prfx));
        return modcfg, bogocfg;


class neko_simple_f2v_aggr_agt_factory(virtual_agt_factory):
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        dataprfx=neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        modprfx =neko_get_arg(this.MOD_PRFX_local, mod_prfx_dict);
        agtcfg = awa.append_agent_to_cfg(agtcfg, AN.MKPROTO(agt_prfx), neko_simple_map_tower_to_vec_aaggr_agent_builder.get_agtcfg(
            VN.I_FEAT_MAP(dataprfx), VN.FEAT_MAP_LST(dataprfx),
            VN.GLOBAL_ATT_MAP(dataprfx), VN.I_FEAT_VEC(dataprfx),
            MN.ATTN(modprfx), "NEP_skipped_NEP"));
        return agtcfg;


class neko_simple_f2v_aagr_part_factory(virtual_part_factory):
    def set_feat2vec_factory(this):
        this.f2vmodf = neko_simple_f2v_aagr_mod_factory(this.gmodcfg, {});
        this.f2vagtf = neko_simple_f2v_aggr_agt_factory();

    def arm_ftwr_fvec_mods(this, modcfg, bogocfg, pipeline_mod_prefix):
        modcfg, bogocfg = this.f2vmodf.arm_module({this.f2vmodf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg);
        return modcfg, bogocfg;
    # if it forks, the forker will fork feature tower--- so do NOT fork implicitly.
    def arm_ftwr_fvec_agts(this, acfg, pipeline_data_prefix, pipeline_mod_prefix):
        # for oscr there is no routing--- hence we
        acfg=this.f2vagtf.arm_agt_core({this.f2vagtf.MOD_PRFX_local: pipeline_mod_prefix},{this.f2vagtf.DATA_PRFX_local:pipeline_data_prefix},acfg,pipeline_data_prefix,{});
        return acfg;

    def setup_factories(this):
        super().setup_factories();
        this.set_feat2vec_factory();
