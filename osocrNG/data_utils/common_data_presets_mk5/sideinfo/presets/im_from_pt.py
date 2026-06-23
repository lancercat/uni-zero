
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5

from osocrNG.data_utils.common_data_presets_mk5.sideinfo.holders.pt_holder import neko_sideinfo_diskloader_pt_factory
from osocrNG.data_utils.common_data_presets_mk5.sideinfo.collators.image_collator import neko_pt_glyph_sinfo_collate_factory
# in theory, we want 32x being able to construct center from any font, but this may not stage in the first revision.
# i need someone to fork myself.
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.var_names import P
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent


class collated_image_sideinfo_from_pt_factory(neko_virtual_factory_mk5):
    MODALITY_NAME="noto_im_sinfo"

    META_PRFX="meta_prefix";
    SIDE_INFO_PRFX="sinfo_prfx"

    PARAM_sinfo_im_size="img_size";
    PARAM_padding_size_hw="padding_size_hw";
    PARAM_sideinfodb_root="sideinforoot";

    HOLDR=neko_sideinfo_diskloader_pt_factory;
    COL=neko_pt_glyph_sinfo_collate_factory;

    @classmethod
    def arm_sideinfo_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        acfg=cls.HOLDR.arm_sideinfoldr(acfg, data_prfxs, mod_prfxs, agt_prfxs, params);
        acfg=cls.COL.arm_collate_mvn_agts(acfg, data_prfxs, mod_prfxs, agt_prfxs, params);
        # this sideinfo config does not want to alter tdict.

        return acfg;


