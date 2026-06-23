from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.libcollate.agents.cv2_pil_collate.image_collate_fix_size_stretch import neko_image_collate_fixed_size_stretch_cv2
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import neko_singlelist_cat_agent

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.data.agents.mvn_dev.neko_modless_mvn_listim import neko_list_rgb_mvn_agent


# well maybe collate it pre aug is better? we will mod the aug stuff if necessary.
# the gist is not to have augmentation work on super large images.
class neko_im_sinfo_collate_factory(neko_virtual_factory_mk5):
    SIDE_INFO_PRFX="sinfo_prfx"
    PARAM_sinfo_im_size="img_size";
    PARAM_padding_size_hw="padding_size_hw";
    @classmethod
    def arm_collate_mvn_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        sinfo_prfx = data_prfxs[cls.SIDE_INFO_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, sinfo_prfx);
        thumbsize = neko_get_arg(cls.PARAM_sinfo_im_size, params);
        padding_size_hw = neko_get_arg(cls.PARAM_padding_size_hw, params)
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "sinfo_collator"),
                                       neko_image_collate_fixed_size_stretch_cv2.get_agtcfg(
                                           VN.IM_raw(sinfo_prfx), VN.IM_tensor_list(sinfo_prfx), thumbsize,
                                           padding_size_hw, -1,
                                       ));
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "sinfo_mvn"),
                                       neko_list_rgb_mvn_agent.get_agtcfg(
                                           VN.IM_tensor_list(sinfo_prfx), VN.IM_normed_tensor_list(sinfo_prfx)
                                       ));
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"sinfo_cat"),neko_singlelist_cat_agent.get_agtcfg(
            VN.IM_normed_tensor_list(sinfo_prfx),VN.IM_normed_tensor_cpu(sinfo_prfx),0));
        return acfg;
# pt
class neko_pt_glyph_sinfo_collate_factory(neko_virtual_factory_mk5):
    SIDE_INFO_PRFX="sinfo_prfx"
    PARAM_sinfo_im_size="img_size";
    PARAM_padding_size_hw="padding_size_hw";
    @classmethod
    def arm_collate_mvn_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        sinfo_prfx = data_prfxs[cls.SIDE_INFO_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, sinfo_prfx);
        thumbsize = neko_get_arg(cls.PARAM_sinfo_im_size, params);
        padding_size_hw = neko_get_arg(cls.PARAM_padding_size_hw, params)
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "sinfo_collator"),
                                       neko_image_collate_fixed_size_stretch_cv2.get_agtcfg(
                                           VN.IM_raw(sinfo_prfx), VN.IM_normed_tensor_cpu(sinfo_prfx), thumbsize,
                                           padding_size_hw, -1,
                                       ));
        return acfg;
