
from neko_sdk.neko_framework_NG.libcollate.modules.neko_fill_gridmker import neko_fill_fixed_size_static


from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN

from neko_2025_NGNW.common.factories.components.collate.grid_collate_af import neko_abstract_grid_collate_mod_factory,neko_grid_based_collate_factory
class neko_strech_and_fill_collate_mod_factory(neko_abstract_grid_collate_mod_factory):
    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
    # well if everything looks normal, just arm everything...
        localprfx=mod_prfx_dict[this.MOD_PRFX_local];
        h,w=neko_get_arg(this.PARAM_template_size_hw,params);
        modcfg=this.config_saveable(modcfg,neko_fill_fixed_size_static,{
            neko_fill_fixed_size_static.PARAM_template_h: h,
            neko_fill_fixed_size_static.PARAM_template_w:w,
        },MN.SAM_COLLATE(localprfx));
        return modcfg,bogocfg;

class neko_stretch_fill_collate_factory(neko_grid_based_collate_factory):
    def set_collate_modf(this):
        this.collate_modf = neko_strech_and_fill_collate_mod_factory(this.gmodcfg);
