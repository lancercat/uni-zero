from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_agent
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_2025_NGNW.common.factories.components.prototyper.sp_injection import neko_sp_injection_mod_factory, \
    neko_sp_injection_agt_factory


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.neko_framework_NG.agents.utils.normalize_at_dim import neko_norm_at_agent
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_part_factory,global_mod_cfg
# prototypes will need to be normalized.
# this sets converts some char features to prototypes--- any char features---
# 32x will come interesting
class neko_as_prototype_factory(virtual_part_factory):
    PARAM_share_mapping="share_mapping";
    def __init__(this, platform_cfg: neko_platform_cfg,gmodcfg:global_mod_cfg, params):
        super().__init__(platform_cfg,gmodcfg,params);
        this.share_mapping=neko_get_arg(this.PARAM_share_mapping,params,{});
    def set_sp_injection_factories(this):
        this.sp_inject_mod = neko_sp_injection_mod_factory(this.gmodcfg, {});
        this.sp_inject_agt = neko_sp_injection_agt_factory();
    def setup_factories(this):
        super().setup_factories()
        this.set_sp_injection_factories();

    def arm_mod_pipeline(this, modcfg, bogocfg,armer_mod_prfx):
        if (armer_mod_prfx not in this.share_mapping):
            modcfg, bogocfg = this.sp_inject_mod.arm_module(
                {this.sp_inject_mod.MOD_PRFX_local:  armer_mod_prfx},
                modcfg, bogocfg, None
            );
        return modcfg,bogocfg;
    def arm_proto_arming_agent(this,acfg,charfeat_data_prfx,armed_data_prfx,armer_mod_prfx):
        if(armer_mod_prfx in this.share_mapping):
            armer_mod_prfx = this.share_mapping[armer_mod_prfx];
        this.sp_inject_agt.arm_agt_core(
            {this.sp_inject_agt.MOD_PRFX_local: armer_mod_prfx},
            {this.sp_inject_agt.DATA_PRFX_wosp: charfeat_data_prfx, this.sp_inject_agt.DATA_PRFX_local: armed_data_prfx,
             this.sp_inject_agt.DATA_PRFX_tdict: charfeat_data_prfx}, acfg, P(armed_data_prfx, "sp_injection")
        );
        acfg=awa.append_agent_to_cfg(acfg,"norm", neko_norm_at_agent.get_agtcfg(
            VN.I_FEAT_VEC(armed_data_prfx),VN.PROTO(armed_data_prfx),-1
        ));
        return acfg