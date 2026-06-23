# image as side info, and re
from neko_sdk.neko_framework_NG.libcollate.agents.id_collate import neko_id_collate

# collate template images, make plabel. The unk will be armed on the fly--- s
# in 32x, each model decide what they are gonna do with unknown--
# ignore, reject, or align to some high entropy distribution
# and hence, it should not be put here....
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

class neko_basic_id_only_meta_collator_factory:
    @classmethod
    def get_meta_collator(cls,in_sideinfo_raw,
                          out_sideinfo_tensor_id):
        acfg=awa.empty();
        acfg=awa.append_agent_to_cfg(acfg,"side_info_collate",
                neko_id_collate.get_agtcfg(
                    in_sideinfo_raw,out_sideinfo_tensor_id
                ));
        # plabel will be made in the classifer. nothing big, don't decorate,

        return acfg;
