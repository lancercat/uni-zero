# not all stuff appear in your model,
# but if it appears, it goes with the listed name.
class basic_data_info:
    IMAGE = "image";
    LABEL="label";
    UID="descp"; # the unique descriptor of a image.

class raw_data_item_names(basic_data_info):
    PREAUG="preaug";
    BEACON="beacon";
    SIZE="size";
    ANCHOR="anchor";
    META_DICT="meta_dict";
    @staticmethod
    def uid2str(uid):
        return "-".join([str(uid[k]) for k in uid]);


# mk 2 adds some things that are data oriented.
# you know them (or not) from dataset.
# we will have the main meta here, but may get ignored after 33x
# page/ line ocr will be in mk5. (object 33x)
# note these are strictly "DATA" oriented...
class basic_data_info_mk2(basic_data_info):
    BMASK="bmask";
    IMASK = "imask";
    SMASK="smask";
    CBOX="chrltrb"
    ROTS="rotation_prior";
    LANG="lang";
    # yes... augment with box will be a pain...
    # we will think about that later.
    MAIN_META="meta";
    SRC_NANE="src_name"; # this will serve as a synthetic data identifier.
    @classmethod
    def get_keys(cls):
        return [cls.UID,cls.SRC_NANE,cls.IMAGE,cls.LABEL,cls.LANG,cls.BMASK,cls.CBOX,cls.ROTS,cls.IMASK,cls.MAIN_META]
    @classmethod
    def is_synth(cls,sd):
        return sd[cls.SRC_NANE][:5]=="synth";

# meta names
class meta_src_key:
    META_tdict = "tdict";
    META_utf_list = "utf_list";
    META_side_info_keys = "side_info_keys";
    META_side_info_dict = "side_info_dict";
