

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,global_mod_cfg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from typing import Tuple, List, Dict, Optional,Any
from osocrNG.modular_agents_ocrNG.aggrate_agents.flatten_over_long_edge_mk2 import \
     neko_flatten_against_long_edge_agent_mk2_naive

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory,global_mod_cfg
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk2 import spatial_attention_NG_mk2

# let me be clear the prediction adn reward does NOT happen here, as we now have multi meta sources
class squeeze_short_edge_modf(virtual_mod_factory):

    PARAM_maxT = "maxT";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict:Dict[str,Any]=None):
        super().__init__(gmodcfg,external_factory_dict);

    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        return modcfg,bogocfg;
# maybe in 32x we can remove the l part to head?
class squeeze_short_edge_agtf(virtual_agt_factory):
    def arm_agt_core(this,mod_prfx_dict,data_prfx_dict,agtcfg,agt_prfx,params=None):
        # yields predicted length based decode.
        head_data_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);
        # head_mod_prfx=neko_get_arg(this.MOD_PRFX_local,mod_prfx_dict);
        # we keep them here as we may need them for later attention based aggr.
        lagtcfg=awa.empty();

        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"squeeze",
                                        neko_flatten_against_long_edge_agent_mk2_naive.get_agtcfg(
                                            VN.FEAT_MAP_LST(head_data_prfx),
                                            VN.ROI_FEAT_SEQ(head_data_prfx),

                                        ));
        agt_prfx=awa.append_agent_to_cfg(agtcfg,AN.ATT_AGGR(agt_prfx),lagtcfg);

        return agt_prfx;
class neko_simple_f2seq_sqeeze_short_part_factory(virtual_part_factory):
    def setup_factories(this):
        this.f2smodf = squeeze_short_edge_modf(this.gmodcfg, {});
        this.f2sagtf = squeeze_short_edge_agtf();

    def arm_ftwr_fseq_mods(this, modcfg, bogocfg, pipeline_mod_prefix):
        modcfg, bogocfg = this.f2smodf.arm_module({this.f2smodf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg);
        return modcfg, bogocfg;
    # if it forks, the forker will fork feature tower--- so do NOT fork implicitly.
    def arm_ftwr_fseq_agts(this, acfg, inst_feat_data_prefix, pipeline_mod_prefix):
        # for oscr there is no routing--- hence we
        acfg=this.f2sagtf.arm_agt_core({this.f2sagtf.MOD_PRFX_local: pipeline_mod_prefix},{
            this.f2sagtf.DATA_PRFX_local:inst_feat_data_prefix},acfg,inst_feat_data_prefix,{});
        return acfg;

