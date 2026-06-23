from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_data_factory_mk5
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.libcollate.agents.cv2_pil_collate.image_collate_tensor_list_max_area import (
    neko_image_collate_max_area_align_pil,neko_image_collate_max_area_align_cv2,
    neko_mask_list_collate_max_area_align_cv2,neko_mask_collate_max_area_align_cv2,neko_bin_mask_list_collate_max_area_align_cv2);

from neko_sdk.neko_framework_NG.libcollate.agents.cv2_pil_collate.image_collate_fix_size_stretch import \
    neko_image_collate_fixed_size_stretch_pil, neko_image_collate_fixed_size_stretch_cv2
from neko_sdk.neko_framework_NG.data.agents.mvn_dev.neko_modless_mvn_listim import neko_list_rgb_mvn_agent
from neko_sdk.neko_framework_NG.data.agents.mkid import neko_id_maker_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

# well maybe collate it pre aug is better? we will mod the aug stuff if necessary.
# the gist is not to have augmentation work on super large images.
# this is tied hard to the data names so yeah, we need to repeat that in the meta section
# note that tokenizers are armed with data loader, not collate loader.
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import neko_singlelist_cat_agent

class neko_im_text_collate_factory(neko_virtual_data_factory_mk5):
    PARAM_img_size_max_area="img_size_max_area";
    PARAM_padding_size_hw="padding_size_hw";
    PARAM_thumb_size_hw="thumb_size_hw";
    PARAM_is_np="is_np";
    @classmethod
    def arm_thumbnail_collate_mvn(cls,acfg,data_prfxs,mod_prfxs,agt_prfxs,params):
        data_prefix = data_prfxs[cls.DATA_PRFX]; # well if you have an augmentation and implemented badly...
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        thumbsize = neko_get_arg(cls.PARAM_thumb_size_hw, params);
        padding_size_hw = neko_get_arg(cls.PARAM_padding_size_hw, params)
        is_np=neko_get_arg(cls.PARAM_is_np,params,False);
        if(is_np):
            acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "thumb_collator"),
                                           neko_image_collate_fixed_size_stretch_cv2.get_agtcfg(
                                               VN.IM_raw(data_prefix), VN.SAM_THUMB(data_prefix), thumbsize,
                                               padding_size_hw, -1,
                                           ));
        else:
            acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "thumb_collator"),
                                           neko_image_collate_fixed_size_stretch_pil.get_agtcfg(
                                               VN.IM_raw(data_prefix), VN.SAM_THUMB(data_prefix), thumbsize,
                                               padding_size_hw, -1,
                                           ));
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "thumb_mvn"),
                                       neko_list_rgb_mvn_agent.get_agtcfg(
                                           VN.SAM_THUMB(data_prefix), VN.SAM_THUMB_normed_tensor_list(data_prefix)
                                       ));
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"thumb_cat"),neko_singlelist_cat_agent.get_agtcfg(
            VN.SAM_THUMB_normed_tensor_list(data_prefix),VN.SAM_THUMB_normed_tensor_cpu(data_prefix),0
        )); # even if we are to use some fancier magic later, its for later.
        return acfg;



    @ classmethod
    def arm_collate_mvn(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        img_size_max_area = neko_get_arg(cls.PARAM_img_size_max_area, params);
        padding_size_hw = neko_get_arg(cls.PARAM_padding_size_hw, params);
        is_np=neko_get_arg(cls.PARAM_is_np,params,False);

        if(is_np):
            acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "img_collator"),
                                           neko_image_collate_max_area_align_cv2.get_agtcfg(
                                               VN.IM_raw(data_prefix), VN.SAM_IM(data_prefix), img_size_max_area,
                                               padding_size_hw, -1,
                                           ));
        else:
            acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "img_collator"),
                                           neko_image_collate_max_area_align_pil.get_agtcfg(
                                               VN.IM_raw(data_prefix), VN.SAM_IM(data_prefix), img_size_max_area,
                                               padding_size_hw, -1,
                                           ));
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "img_mvn"),
                                       neko_list_rgb_mvn_agent.get_agtcfg(
                                           VN.SAM_IM(data_prefix), VN.IM_normed_tensor_list(data_prefix)
                                       ));
        acfg=awa.append_agent_to_cfg(acfg, P(aprfx,"mkid"),
                                     neko_id_maker_agent.get_agtcfg(
                                         VN.IM_normed_tensor_list(data_prefix),
                                         VN.SAM_ID(data_prefix)
                                     ));

        return acfg;
    # after mvn the samples are moved to tensor and shipped to the queue.


    @classmethod
    def arm_collates(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);

        col=awa.empty([VN.IM_raw(data_prefix)]);
        col=cls.arm_collate_mvn(col, data_prfxs, mod_prfxs, agt_prfxs, params);
        col=cls.arm_thumbnail_collate_mvn(col, data_prfxs, mod_prfxs, agt_prfxs, params);
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"collate"),col);
        return acfg;

class neko_semins_collate_factory(neko_virtual_data_factory_mk5):
    PARAM_img_size_max_area="img_size_max_area";
    PARAM_padding_size_hw="padding_size_hw";
    PARAM_thumb_size_hw="thumb_size_hw";
    PARAM_is_np="is_np";
    @classmethod
    def arm_collate_msk_lst_core(cls,acfg, img_size_max_area, padding_size_hw,aprfx,iname,oname):
        lacfg=awa.empty([iname]); # only engage if the dataset supports it
        name=P(P(aprfx, "msklst_collator"),iname);
        lacfg = awa.append_agent_to_cfg(lacfg, name,
                                       neko_bin_mask_list_collate_max_area_align_cv2.get_agtcfg(
                                           iname, oname, img_size_max_area,
                                           padding_size_hw, -1,
                                       ));
        return awa.append_agent_to_cfg(acfg,P(aprfx,iname),lacfg);

    @classmethod
    def arm_collate_msk_core(cls,acfg, img_size_max_area, padding_size_hw,aprfx,iname,oname):
        lacfg=awa.empty([iname]);
        name=P(P(aprfx, "msk_collator"),iname);
        lacfg = awa.append_agent_to_cfg(lacfg, name,
                                       neko_mask_collate_max_area_align_cv2.get_agtcfg(
                                           iname, oname, img_size_max_area,
                                           padding_size_hw, -1,
                                       ));
        return awa.append_agent_to_cfg(acfg,P(aprfx,iname),lacfg);

    # so
    @classmethod
    def arm_collate_mask_optional(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[cls.DATA_PRFX];
        col=awa.empty([VN.IM_raw(data_prefix)]);
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        img_size_max_area = neko_get_arg(cls.PARAM_img_size_max_area, params);
        padding_size_hw = neko_get_arg(cls.PARAM_padding_size_hw, params);

        acfg = cls.arm_collate_msk_core(acfg, img_size_max_area, padding_size_hw, aprfx, VN.GT_BIN_MSK(data_prefix),
                                        VN.GT_BIN_MSK_TEN(data_prefix))
        # collate semseg if dataset provides it
        acfg = cls.arm_collate_msk_lst_core(acfg, img_size_max_area, padding_size_hw, aprfx,
                                            VN.GT_SEMSEG_CBM_MSK(data_prefix), VN.GT_SEMSEG_CBM_MSK_TEN(data_prefix));
        # collate insseg if dataset provides it
        acfg = cls.arm_collate_msk_lst_core(acfg, img_size_max_area, padding_size_hw, aprfx,
                                            VN.GT_INSSEG_MSK(data_prefix), VN.GT_INSSEG_DENSE_MSK_TEN_LST(data_prefix));

        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"msk_collate"),col);
        return acfg;