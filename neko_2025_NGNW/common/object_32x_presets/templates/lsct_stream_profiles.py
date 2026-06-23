import torch

from neko_sdk.neko_framework_NG.names import P
import os
from neko_2025_NGNW.common.object_32x_presets.templates.data_diskcol_profile import neko_diskcol_template
class neko_lsct_template:
    NAME = "name";
    TYPE="type";

    DATA_ENDPOINTS = "data_endpoints";
    META_ENDPOINTS = "meta_endpoints";
    DSKCOL_TEMPLATE="col_template";
    BG_DB = "bg_db";
    ORIENTS="orients"
    SPACE="space";
    FNT_SIZE = "fnt_size";


class neko_lsct_sample_grounded_template(neko_lsct_template):
    REF_DATA_PRFX = "ref_data_prfx";
    FNT_ROOT = "FNT_ROOT";
    FNT_IDX_PATH = "font_db";

    @classmethod
    def cfg_core(cls, ref_data_prfx, data_root, name,type, fsz=32,spacing=None):
        if(spacing is None):
            spacing=[0, 32, 64];
        return {
            cls.DSKCOL_TEMPLATE: neko_diskcol_template.get_default_diskcol_template(),
            cls.TYPE: type,
            cls.FNT_IDX_PATH: os.path.join(data_root, "synth_lsct/cache/finfo251105.pt"),
            cls.BG_DB: os.path.join(data_root, "synth_lsct/bg_img"),
            cls.FNT_ROOT: os.path.join(data_root, "synth_lsct/"),
            cls.FNT_SIZE: fsz,
            cls.REF_DATA_PRFX: ref_data_prfx,
            cls.ORIENTS: [0, 1],
            cls.SPACE: spacing,
            cls.DATA_ENDPOINTS: [name],
            cls.META_ENDPOINTS: [],  # shadow lsct impls do not create meta stream
        }
# LSCT is so diverse. Let's use different template to hold different params
class neko_lsct_repeater_template(neko_lsct_sample_grounded_template):
    THIS_TYPE = "lsct_repeater";  # there is no need for meta repeater--- simply doesn't make sense.
    @classmethod
    def get_default_lsct_repeater(cls,ref_data_prfx,data_root,name):
        return cls.cfg_core(ref_data_prfx,data_root,name,cls.THIS_TYPE);
# just fork the stream
class neko_lsct_stream_fork_template(neko_lsct_repeater_template):
    THIS_TYPE="lsct_repeater"; # there is no need for meta repeater--- simply doesn't make sense.
    @classmethod
    def get_default_lsct_repeater(cls,ref_data_prfx,data_root,name,fsz=32,spacing=None):
        cfg= cls.cfg_core(ref_data_prfx,data_root,name,cls.THIS_TYPE,fsz,spacing);
        cfg[cls.ORIENTS]=[0];
        return cfg;
class neko_lsct_ho_repeater_template(neko_lsct_repeater_template):
    THIS_TYPE="lsct_repeater"; # there is no need for meta repeater--- simply doesn't make sense.
    @classmethod
    def get_default_lsct_repeater(cls,ref_data_prfx,data_root,name,fsz=32,spacing=None):
        cfg= cls.cfg_core(ref_data_prfx,data_root,name,cls.THIS_TYPE,fsz,spacing);
        cfg[cls.ORIENTS]=[0];
        return cfg;
class neko_lsct_ho_shuff_template(neko_lsct_sample_grounded_template):
    THIS_TYPE = "lsct_shuff";  # there is no need for meta repeater--- simply doesn't make sense.
    @classmethod
    def get_default_lsct_shuff(cls,ref_data_prfx,data_root,name,fsz=32,spacing=None):
        cfg= cls.cfg_core(ref_data_prfx,data_root,name,cls.THIS_TYPE,fsz,spacing);
        cfg[cls.ORIENTS]=[0];
        return cfg;

class neko_lsct_char_random_template(neko_lsct_template):
    THIS_TYPE="lsct_fullrand";
    CHARSET="charset";
    BATCH_SIZE="batchsize"
    DB_PATH="lmdb_path";
    @classmethod
    def get_default_lsct_rand_char_core(cls, charset,db_path, data_root, name,batchsize,fsz=32):
        return {
            cls.DSKCOL_TEMPLATE: neko_diskcol_template.get_default_diskcol_template(),
            cls.TYPE: cls.THIS_TYPE,
            cls.DB_PATH: os.path.join(data_root, db_path),
            cls.BG_DB: os.path.join(data_root, "synth_lsct/bg_img"),
            cls.CHARSET: charset,
            cls.ORIENTS: [0],
            cls.FNT_SIZE: fsz,
            cls.BATCH_SIZE: batchsize,
            cls.DATA_ENDPOINTS: [P(name,"data")],
            cls.META_ENDPOINTS: [P(name,"meta")],
        }
    @classmethod
    def get_default_lsct_rand_char(cls, charset, data_root, name,batchsize,fsz=32):
        db_path= "synth_lsct/glyphdb_utf";
        return cls.get_default_lsct_rand_char_core(charset,db_path, data_root, name,batchsize,fsz);
    @classmethod
    def get_default_lsct_rand_charXL(cls, charset, data_root, name,batchsize,fsz=32):
        db_path= "synth_lsct/glyphdb_utf_260202";
        return cls.get_default_lsct_rand_char_core(charset,db_path, data_root, name,batchsize,fsz);
