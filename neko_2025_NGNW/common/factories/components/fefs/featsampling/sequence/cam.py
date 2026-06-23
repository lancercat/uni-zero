
from osocrNG.modular_agents_ocrNG.aggrate_agents.feat_aggr_agents_mk4 import neko_word_aggr_mk4
# seq flatten is done with prediction and head where they take in length.
# its not needed in non-terminal states either way.
from osocrNG.modular_agents_ocrNG.att_agents.temporal_att_agent import neko_basic_temporal_attention
from osocrNG.ocr_modules_NG.temporal_att.attention_conv_sigmoid_mk1 import attention_conv_sigmoid_mk1
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,global_mod_cfg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from osocrNG.ocr_modules_NG.sampler_NG.dtd_ng_mk2 import neko_DTDNG_mk2
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from typing import Tuple, List, Dict, Optional,Any


# let me be clear the prediction adn reward does NOT happen here, as we now have multi meta sources
class cam_factory_mk2_modf(virtual_mod_factory):
    temporal_attention_engine = attention_conv_sigmoid_mk1;

    PARAM_maxT = "maxT";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict:Dict[str,Any]=None):
        super().__init__(gmodcfg,external_factory_dict);
        this.tfe_channels=this.gmodcfg.tfe_chs;
        this.nparts=this.gmodcfg.nparts;

    def config_dtd(this, modcfgdict,name):
        return this.config_saveable(modcfgdict,neko_DTDNG_mk2,{
        },name);
    def config_attn(this,modcfgdict,name,maxT):
        e=this.temporal_attention_engine;
        return this.config_saveable(modcfgdict,e, {
            e.PARAM_n_parts:this.nparts,
            e.PARAM_maxT:maxT,
            e.PARAM_number_channels:this.tfe_channels},name);


    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        maxT=neko_get_arg(this.PARAM_maxT,params,32);
        modprfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_attn(modcfg,MN.WORD_ATTN(modprfx),maxT);
        modcfg=this.config_dtd(modcfg,MN.WORD_AGGR(modprfx));
        return modcfg,bogocfg;

# maybe in 32x we can remove the l part to head? 
class cam_factory_mk2_agtf(virtual_agt_factory):
    DATA_PRFX_feat="feat_prfx";
    DATA_PRFX_temp="feat_temp";

    def arm_agt_core(this,mod_prfx_dict,data_prfx_dict,agtcfg,agt_prfx,params=None):

        # yields predicted length based decode.
        feat_data_prefix=neko_get_arg(this.DATA_PRFX_feat,data_prfx_dict);
        feat_temp_prefix=neko_get_arg(this.DATA_PRFX_temp,data_prfx_dict);

        head_data_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);
        head_mod_prfx=neko_get_arg(this.MOD_PRFX_local,mod_prfx_dict);
        lagtcfg=awa.empty();

        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"attn",neko_basic_temporal_attention.get_agtcfg(
                    VN.WORD_TEMP_FEAT_MAP_LAST(feat_temp_prefix),
                    VN.WORD_ATT_MAP(feat_temp_prefix),
                    MN.WORD_ATTN(head_mod_prfx)));
        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"attn_aggr",
            neko_word_aggr_mk4.get_agtcfg(
                VN.I_FEAT_MAP(feat_data_prefix),
                VN.WORD_ATT_MAP(feat_temp_prefix),
                VN.ROI_FEAT_SEQ(head_data_prfx),
            MN.WORD_AGGR(head_mod_prfx)));

        agt_prfx=awa.append_agent_to_cfg(agtcfg,AN.ATT_AGGR(agt_prfx),lagtcfg);

        return agt_prfx;
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory

class neko_simple_f2seq_attn_part_factory(virtual_part_factory):
    def setup_factories(this):
        this.f2smodf = cam_factory_mk2_modf(this.gmodcfg, {});
        this.f2sagtf = cam_factory_mk2_agtf();

    def arm_ftwr_fseq_mods(this, modcfg, bogocfg, pipeline_mod_prefix):
        modcfg, bogocfg = this.f2smodf.arm_module({this.f2smodf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg);
        return modcfg, bogocfg;
    # if it forks, the forker will fork feature tower--- so do NOT fork implicitly.
    def arm_ftwr_fseq_agts(this, acfg, inst_feat_data_prefix,temporal_feat_data_prefix, pipeline_mod_prefix):
        # for oscr there is no routing--- hence we
        acfg=this.f2sagtf.arm_agt_core(
            {this.f2sagtf.MOD_PRFX_local: pipeline_mod_prefix},
            {this.f2sagtf.DATA_PRFX_feat:inst_feat_data_prefix,
             this.f2sagtf.DATA_PRFX_temp:temporal_feat_data_prefix,
             this.f2sagtf.DATA_PRFX_local:inst_feat_data_prefix,
             },acfg,inst_feat_data_prefix,{});
        return acfg;

