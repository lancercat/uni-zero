# image as side info, and re
from neko_sdk.neko_framework_NG.libcollate.agents.cv2_pil_collate.image_collate_fix_max_edge import neko_image_collate_fixed_size_max_edge_align_cv2

# collate template images, make plabel. The unk will be armed on the fly--- s
# in 32x, each model decide what they are gonna do with unknown--
# ignore, reject, or align to some high entropy distribution
# and hence, it should not be put here....
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

class neko_basic_image_only_meta_collator_factory:
    @classmethod
    def get_meta_collator(cls,in_sideinfo_im,
                          out_sideinfo_tensor_im,template_size_hw):
        acfg=awa.empty();
        acfg=awa.append_agent_to_cfg(acfg,"side_info_collate",
                neko_image_collate_fixed_size_max_edge_align_cv2.get_agtcfg(
                    in_sideinfo_im,out_sideinfo_tensor_im,template_size_hw,(2,2),0
                ));
        return acfg;