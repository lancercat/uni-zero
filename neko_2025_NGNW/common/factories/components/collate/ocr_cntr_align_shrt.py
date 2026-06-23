from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN

from neko_sdk.neko_framework_NG.libcollate.modules.neko_ocr_cntr_algn_shrt_gridmker import neko_ocr_cntr_algn_shrt
from neko_2025_NGNW.common.factories.components.collate.grid_collate_af import neko_abstract_grid_collate_mod_factory,neko_grid_based_collate_factory

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory,global_mod_cfg
class neko_ocr_cntr_align_collate_mod_factory(neko_abstract_grid_collate_mod_factory):
    PARAM_template_size_hw="template_hw";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(gmodcfg,external_factory_dict)


    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
    # well if everything looks normal, just arm everything...
        localprfx=mod_prfx_dict[this.MOD_PRFX_local];
        h,w=neko_get_arg(this.PARAM_template_size_hw,params);
        modcfg=this.config_saveable(modcfg,neko_ocr_cntr_algn_shrt,{
            neko_ocr_cntr_algn_shrt.PARAM_template_h: h,
            neko_ocr_cntr_algn_shrt.PARAM_template_w:w,
        },MN.SAM_COLLATE(localprfx));
        return modcfg,bogocfg;



class neko_ocr_cntr_algn_factory(neko_grid_based_collate_factory):
    def set_collate_modf(this):
        this.collate_modf = neko_ocr_cntr_align_collate_mod_factory(this.gmodcfg);

