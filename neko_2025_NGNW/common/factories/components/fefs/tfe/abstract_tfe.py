from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.log import fatal
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_mod_factory,virtual_agt_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from osocrNG.ocr_modules_NG.spatial_embedding.se_dino_like import temporal_se_dino_like_mk1
from osocrNG.ocr_modules_NG.temporal_fe.temporal_fe_fpn import temporal_fe_fpn, temporal_fe_fpn_supres
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from osocrNG.configs.typical_agent_setups.detached_se_fpn import get_detached_se_fpn,get_attached_se_fpn
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import neko_pool_and_cat_agent
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_simple_action_module_wrapping_agent_1i1o_basic
from neko_sdk.neko_framework_NG.modules.mlp import neko_normed_mlp
class abstract_tfe_mod_factory(virtual_mod_factory):
    spatial_embedding_engine=temporal_se_dino_like_mk1;
    temporal_fe_engine=temporal_fe_fpn;


    def __init__(this,gmodcfg, external_factory_dict):
        super().__init__(gmodcfg,external_factory_dict);
        this.tfe_channels=this.gmodcfg.tfe_chs;

    def stopper_channels(this):
        return -1;


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
        fatal("not impl");
        return modcfgdict;

    def config_temporal_fe_digester(this,modcfgdict,name):
        modcfgdict = this.config_saveable(modcfgdict, neko_normed_mlp, {
            neko_normed_mlp.PARAM_in_dim: this.stopper_channels(),
            neko_normed_mlp.PARAM_out_dim: this.tfe_channels,
        }, name)
        return modcfgdict

    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        prfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg = this.config_spatial_embedding(
            modcfg, MN.WORD_TEMPORAL_SE(prfx));
        modcfg = this.config_temporal_fe(modcfg, MN.WORD_TEMPORAL_FE(prfx));
        modcfg=this.config_temporal_fe_digester(modcfg,MN.WORD_TEMPORAL_GFE(prfx));
        return modcfg,bogocfg;

