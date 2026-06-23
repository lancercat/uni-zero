import torch
from osocrNG.sptokens import tUNK

from neko_2025_NGNW.common.modules.quick_and_ugly_sp import neko_quick_and_dirty_sp
from neko_2025_NGNW.common.agents.quick_and_ugly_sp_injection import neko_inject_sp
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory,global_mod_cfg
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.libmeta.mkplabel import neko_plabel_mker


# gonna split this into two things....
class neko_sp_injection_mod_factory(virtual_mod_factory):
    # bcs you want shared fe. Ofc you can just
    PARAM_sedim="sedim";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(gmodcfg,external_factory_dict);

    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        prfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_saveable(
            modcfg,neko_quick_and_dirty_sp,{
                neko_quick_and_dirty_sp.PARAM_parts:this.gmodcfg.nparts,neko_quick_and_dirty_sp.PARAM_num_ch:this.gmodcfg.feat_ch_model},MN.META_SPINJ(prfx)
        );
        return modcfg,bogocfg;


class neko_sp_injection_agt_factory(virtual_agt_factory):
    DATA_PRFX_tdict="tdict_prfx";
    DATA_PRFX_wosp="wospprfx";

    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict,  agtcfg, agt_prfx, params=None):
        dataprfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        tdictprfx = neko_get_arg(this.DATA_PRFX_tdict, data_prfx_dict);
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        wosp=neko_get_arg(this.DATA_PRFX_wosp, data_prfx_dict);
        lacfg=awa.empty();
        lacfg=awa.append_agent_to_cfg(lacfg,"inject_sp",neko_inject_sp.get_agtcfg(
            VN.I_FEAT_VEC(wosp),VN.TDICT(tdictprfx),VN.UTF(wosp),
            VN.I_FEAT_VEC(dataprfx),VN.TDICT(dataprfx),VN.UTF(dataprfx), # you can load however many templates you wish--- just make sure you match the side_info_utf.
            MN.META_SPINJ(modprfx)
        ));
        lacfg=awa.append_agent_to_cfg(lacfg,P(dataprfx,"plabel-maker"),
           neko_plabel_mker.get_agtcfg(
               VN.I_FEAT_VEC(dataprfx), VN.TDICT(dataprfx), VN.UTF(dataprfx),
               VN.PROTO_LABEL(dataprfx),[tUNK]
        ));
        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.SP_INJECTION(agt_prfx),lacfg);
        return agtcfg;

