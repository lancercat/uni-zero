
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5

from osocrNG.data_utils.common_data_presets_mk5.sideinfo.holders.database_holder import neko_sideinfo_diskloader_lmdb_factory
from osocrNG.data_utils.common_data_presets_mk5.sideinfo.collators.image_collator import neko_pt_glyph_sinfo_collate_factory

class collated_kim_sideinfo_from_lmdb_factory(neko_virtual_factory_mk5):
    MODALITY_NAME="noto_im_sinfo"

    META_PRFX="meta_prefix";
    SIDE_INFO_PRFX="sinfo_prfx"

    PARAM_k="k";
    PARAM_rand="rand";  # if you don't want to turn your testing process into a random process :)
    PARAM_sinfo_im_size="img_size";
    PARAM_padding_size_hw="padding_size_hw";
    PARAM_sideinfodb_root="sideinforoot";

    HOLDR=neko_sideinfo_diskloader_lmdb_factory;
    COL=neko_pt_glyph_sinfo_collate_factory;

    @classmethod
    def arm_sideinfo_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        acfg=cls.HOLDR.arm_sideinfoldr(acfg, data_prfxs, mod_prfxs, agt_prfxs, params);
        acfg=cls.COL.arm_collate_mvn_agts(acfg, data_prfxs, mod_prfxs, agt_prfxs, params);
        # this sideinfo config does not want to alter tdict.

        return acfg;


