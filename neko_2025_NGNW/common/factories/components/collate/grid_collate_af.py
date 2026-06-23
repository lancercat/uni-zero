
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.neko_framework_NG.libcollate.agents.tensor_img_collate.mkgrid_with_ref import (
    neko_make_grid_with_ref)
from neko_sdk.neko_framework_NG.libcollate.agents.tensor_img_collate.grid_sample import neko_image_list_grid_sampler_collate

from neko_sdk.cfgtool.argsparse import neko_get_arg


from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory,global_mod_cfg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
class neko_im_list_collate_agt_factory(virtual_agt_factory):
    # this means the grid maker will compute representation by itself from thumbnail.


    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict,agtcfg,agt_prfx,params=None):
        localdataprfx=data_prfx_dict[this.DATA_PRFX_local];
        localmodprfx=mod_prfx_dict[this.MOD_PRFX_local];
        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.MKGRID(agt_prfx),neko_make_grid_with_ref.get_agtcfg(
            VN.IM_normed_tensor_list(localdataprfx),VN.THUMB_FEATUREMAP_TWR(localdataprfx),VN.THUMB_FEATURE(localdataprfx),VN.COLL_GRID(localdataprfx),MN.SAM_COLLATE(localmodprfx)
        ));
        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.GRIDCOLL(agt_prfx),neko_image_list_grid_sampler_collate.get_agtcfg(
            VN.IM_normed_tensor_list(localdataprfx),VN.COLL_GRID(localdataprfx), VN.IM_normed_tensor(localdataprfx)
        ))
        return agtcfg;





class neko_abstract_grid_collate_mod_factory(virtual_mod_factory):
    PARAM_template_size_hw="template_hw";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(gmodcfg,external_factory_dict)


    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        fatal("not impl");

        return modcfg,bogocfg;
class neko_grid_based_collate_factory(virtual_part_factory):
    PARAM_MOD_imhw="imhw";
    def set_collate_modf(this):
        this.collate_modf = None;
    def setup_factories(this):
        super().setup_factories();
        this.set_collate_modf();
        this.collate_agtf = neko_im_list_collate_agt_factory();
    def arm_listim_devten_mod(this, modcfg, bogocfg, pipeline_mod_prefix, param):
        modcfg,bogocfg = this.collate_modf.arm_module(
            {this.collate_modf.MOD_PRFX_local:pipeline_mod_prefix},
            modcfg,bogocfg,
            {this.collate_modf.PARAM_template_size_hw:neko_get_arg(this.PARAM_MOD_imhw,param)});
        return modcfg,bogocfg;
    def arm_listim_devten_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.collate_agtf.arm_agt_core({this.collate_agtf.MOD_PRFX_local: mod_prfx},
                                              {this.collate_agtf.DATA_PRFX_local: data_prfx}, acfg, data_prfx);
        return acfg;
