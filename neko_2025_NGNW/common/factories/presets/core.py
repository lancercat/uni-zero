# Standard Imports


# Framework / SDK Imports (neko_sdk)
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_2025_NGNW.common.factories.components.backbone.res45dsbnpool import neko_res45dsbn_fe_nf_part_factory
from neko_2025_NGNW.common.object_32x_presets.cfgutil import global_mod_cfg


class neko_object320_factory_core:
    PARAM_mod_prefix = "mprfx";  # global prefix. All agents/moddule produced here are mounted under this prefix, bcs we may eventually need misd.
    # collate should not be handled by fe--- its too customed
    # from cpu batch image tensor  to feature map

    def training_endpoint_prfx(this, prfx):
        return P("training", prfx);

    # this is for future per part override
    def set_bbn_factory(this):
        this.bbnf = neko_res45dsbn_fe_part_factory(this.platform_cfg, this.gmodcfg, {});

    def __init__(this,platform_cfg: neko_platform_cfg, params):
        this.platform_cfg = platform_cfg;
        this.gmodcfg = global_mod_cfg(platform_cfg, params);
        this.global_mod_prefix = neko_get_arg(this.PARAM_mod_prefix, params, "nep");
        this.set_bbn_factory();
class neko_object320_factory_core_ffn(neko_object320_factory_core):

    # this is for future per part override
    def set_bbn_factory(this):
        this.bbnf = neko_res45dsbn_ffn_fe_part_factory(this.platform_cfg, this.gmodcfg, {});

#
class neko_object320_factory_core_nf(neko_object320_factory_core):
    # this is for future per part override
    def set_bbn_factory(this):
        this.bbnf = neko_res45dsbn_fe_nf_part_factory(this.platform_cfg, this.gmodcfg, {});


