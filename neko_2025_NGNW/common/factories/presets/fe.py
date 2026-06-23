
from neko_2025_NGNW.common.factories.components.collate.stretch_and_fill import neko_stretch_fill_collate_factory


from neko_2025_NGNW.common.factories.components.collate.ocr_cntr_align_shrt import neko_ocr_cntr_algn_factory
from neko_2025_NGNW.common.factories.presets.core import neko_object320_factory_core

class neko_object320_fef_abstract:
    def set_collate_factories(this):
        pass;
    def __init__(this, core:neko_object320_factory_core):
        this.core=core;
        this.set_collate_factories();
    def arm_listim_ftwr_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        fatal("not impl");
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.
    def arm_listim_ftwr_agt(this, acfg, data_prfx, mod_prfx):
        fatal("not impl");
        return acfg;

class neko_object320_fef_fill(neko_object320_fef_abstract):
    def set_collate_factories(this):
        this.fscol=neko_stretch_fill_collate_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    def arm_listim_ftwr_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fscol.arm_listim_devten_mod(modcfg,bogocfg,pipeline_mod_prefix,{this.fscol.PARAM_MOD_imhw:imhw});
        modcfg,bogocfg=this.core.bbnf.arm_devten_ftwr_mods(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_ftwr_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.fscol.arm_listim_devten_agt(acfg,data_prfx,mod_prfx);
        acfg=this.core.bbnf.arm_devten_ftwr_agts(acfg,data_prfx, mod_prfx);
        return acfg;
class neko_object320_fef_cntrocr(neko_object320_fef_fill):
    def set_collate_factories(this):
        this.fscol=neko_ocr_cntr_algn_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    def arm_listim_ftwr_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fscol.arm_listim_devten_mod(modcfg,bogocfg,pipeline_mod_prefix,{this.fscol.PARAM_MOD_imhw:imhw});
        modcfg,bogocfg=this.core.bbnf.arm_devten_ftwr_mods(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_ftwr_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.fscol.arm_listim_devten_agt(acfg,data_prfx,mod_prfx);
        acfg=this.core.bbnf.arm_devten_ftwr_agts(acfg,data_prfx, mod_prfx);
        return acfg;

    pass;