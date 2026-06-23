from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.names import P

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.agents.sideinfo_repos.utf_im_ram_sideinfo import utf_precached_im_ram_sideinfo_one
# well maybe collate it pre aug is better? we will mod the aug stuff if necessary.
# the gist is not to have augmentation work on super large images.
class neko_sideinfo_diskloader_pt_factory(neko_virtual_factory_mk5):
    META_PRFX="meta_prefix";
    SIDE_INFO_PRFX="sinfo_prfx";
    PARAM_sideinfodb_root="sideinforoot";
    @classmethod
    def arm_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        meta_prfx = data_prfxs[cls.META_PRFX];
        sinfo_prfx = data_prfxs[cls.SIDE_INFO_PRFX];

        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, meta_prfx);
        # usually tdict shall not be adapted by sideinfo loader--- but if we need to process are Homograph in some modality then why not....
        # not like tdicts are large either way :-)
        acfg=awa.append_agent_to_cfg(
            acfg,P(aprfx,"meta_ldr"),
            utf_precached_im_ram_sideinfo_one.get_agtcfg(
                VN.UTF(meta_prfx),VN.TDICT(meta_prfx),
                VN.IM_raw(sinfo_prfx),VN.UTF(sinfo_prfx),VN.TDICT(sinfo_prfx),
                params[cls.PARAM_sideinfodb_root]
        ));

        return acfg

