from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_mod_factory,virtual_agt_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from osocrNG.ocr_modules_NG.spatial_embedding.se_dino_like import temporal_se_dino_like_mk1
from osocrNG.ocr_modules_NG.temporal_fe.temporal_fe_fpn import temporal_fe_fpn
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from osocrNG.configs.typical_agent_setups.detached_se_fpn import get_detached_se_fpn,get_attached_se_fpn
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import neko_pool_and_cat_agent
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_simple_action_module_wrapping_agent_1i1o_basic
from neko_sdk.neko_framework_NG.modules.mlp import neko_normed_mlp
class neko_danish_tfe_mod_factory(virtual_mod_factory):
    spatial_embedding_engine=temporal_se_dino_like_mk1;
    temporal_fe_engine=temporal_fe_fpn;


    def __init__(this,gmodcfg, external_factory_dict):
        super().__init__(gmodcfg,external_factory_dict);
        this.cam_channels=this.gmodcfg.tfe_chs;
        this.stopper_channels=this.cam_channels*5;
        this.CAM_drop=0.5;


    def config_spatial_embedding(this,modcfgdict,name):
        e=this.spatial_embedding_engine
        modcfgdict=this.config_saveable(
            modcfgdict,e,{
                e.PARAM_scales:  [
                    [this.gmodcfg.fe_ochs[1] + this.gmodcfg.num_se_channels, 16, 64],
                    [this.gmodcfg.fe_ochs[3] + this.gmodcfg.num_se_channels, 8, 32],
                    [this.gmodcfg.fe_ochs[-1] + this.gmodcfg.num_se_channels, 8, 32]
                ],
                e.PARAM_num_se_channels: this.gmodcfg.num_se_channels,
            },name
        )
        return modcfgdict;
    def config_temporal_fe(this,modcfgdict,name):
        e= this.temporal_fe_engine;
        modcfgdict=this.config_saveable(modcfgdict, e, {
            e.PARAM_scales: [
                [this.gmodcfg.fe_ochs[1] + this.gmodcfg.num_se_channels, 16, 64],
                [this.gmodcfg.fe_ochs[3] + this.gmodcfg.num_se_channels, 8, 32],
                [this.gmodcfg.fe_ochs[-1] + this.gmodcfg.num_se_channels, 8, 32]
            ],
            e.PARAM_depth:8,
            e.PARAM_num_se_channels:this.gmodcfg.num_se_channels,
            e.PARAM_num_channels:this.cam_channels
            }, name);
        return modcfgdict
    def config_temporal_fe_digester(this,modcfgdict,name):
        modcfgdict = this.config_saveable(modcfgdict, neko_normed_mlp, {
            neko_normed_mlp.PARAM_in_dim: this.stopper_channels,
            neko_normed_mlp.PARAM_out_dim: this.cam_channels,
        }, name)
        return modcfgdict

    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        prfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg = this.config_spatial_embedding(
            modcfg, MN.WORD_TEMPORAL_SE(prfx));
        modcfg = this.config_temporal_fe(modcfg, MN.WORD_TEMPORAL_FE(prfx));
        modcfg=this.config_temporal_fe_digester(modcfg,MN.WORD_TEMPORAL_GFE(prfx));
        return modcfg,bogocfg;

class neko_danish_tfe_agt_factory(virtual_agt_factory):
    DATA_PRFX_feat="feat_data_prfx";

    def get_agt_prfx(this,mod_prfx_dict,data_prfx_dict,agt_prfx=None):
        if(agt_prfx is None):
            agt_prfx=P(data_prfx_dict[this.DATA_PRFX_local],mod_prfx_dict[this.MOD_PRFX_local]);
        return agt_prfx;

    def arm_agt_core(this,mod_prfx_dict,data_prfx_dict,agtcfg,agt_prfx,params=None):
        aprfx=this.get_agt_prfx(mod_prfx_dict,data_prfx_dict, agt_prfx);
        fprfx=data_prfx_dict[this.DATA_PRFX_feat];
        hdprfx=data_prfx_dict[this.DATA_PRFX_local];

        lmprfx=mod_prfx_dict[this.MOD_PRFX_local];

        lagtcfg=awa.empty();

        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"temproal_fe",
            get_detached_se_fpn(VN.FEAT_MAP_LST(fprfx),
                       VN.WORD_FEAT_MAP_LIST_DETACHED(hdprfx),
                       VN.WORD_FEAT_MAP_LIST_DETACHED_SE(hdprfx),
                       VN.WORD_TEMP_FEAT_MAP_LAST(hdprfx),
                       VN.WORD_TEMP_FEAT_MAP_LIST(hdprfx),
                       MN.WORD_TEMPORAL_SE(lmprfx),
                       MN.WORD_TEMPORAL_FE(lmprfx))
            );
        lagtcfg=awa.append_agent_to_cfg(lagtcfg, "endpoint_pooling", neko_pool_and_cat_agent.get_agtcfg(
            VN.WORD_TEMP_FEAT_MAP_LIST(hdprfx),
            VN.WORD_TEMP_FEAT_GPOOL(hdprfx)
        ));
        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"endpoint_mlp",neko_simple_action_module_wrapping_agent_1i1o_basic.get_agtcfg(
            VN.WORD_TEMP_FEAT_GPOOL(hdprfx),
            VN.WORD_TEMP_FEAT_GLOBAL(hdprfx),
            MN.WORD_TEMPORAL_GFE(lmprfx)
        ));

        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.TFE(aprfx),lagtcfg);
        return agtcfg;
class neko_danish_tfe_GT_agt_factory(neko_danish_tfe_agt_factory):


    def arm_agt_core(this,mod_prfx_dict,data_prfx_dict,agtcfg,agt_prfx,params=None):
        aprfx=this.get_agt_prfx(mod_prfx_dict,data_prfx_dict, agt_prfx);
        fprfx=data_prfx_dict[this.DATA_PRFX_feat];
        hdprfx=data_prfx_dict[this.DATA_PRFX_local];

        lmprfx=mod_prfx_dict[this.MOD_PRFX_local];

        lagtcfg=awa.empty();

        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"temproal_fe",
            get_attached_se_fpn(VN.FEAT_MAP_LST(fprfx),
                       VN.FEAT_MAP_LST_SE(hdprfx),
                       VN.WORD_TEMP_FEAT_MAP_LAST(hdprfx),
                       VN.WORD_TEMP_FEAT_MAP_LIST(hdprfx),
                       MN.WORD_TEMPORAL_SE(lmprfx),
                       MN.WORD_TEMPORAL_FE(lmprfx))
            );
        lagtcfg=awa.append_agent_to_cfg(lagtcfg, "endpoint_pooling", neko_pool_and_cat_agent.get_agtcfg(
            VN.WORD_TEMP_FEAT_MAP_LIST(hdprfx),
            VN.WORD_TEMP_FEAT_GPOOL(hdprfx)
        ));
        lagtcfg=awa.append_agent_to_cfg(lagtcfg,"endpoint_mlp",neko_simple_action_module_wrapping_agent_1i1o_basic.get_agtcfg(
            VN.WORD_TEMP_FEAT_GPOOL(hdprfx),
            VN.WORD_TEMP_FEAT_GLOBAL(hdprfx),
            MN.WORD_TEMPORAL_GFE(lmprfx)
        ));

        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.TFE(aprfx),lagtcfg);
        return agtcfg;