from neko_sdk.neko_framework_NG.names import P
import os
from neko_2025_NGNW.common.object_32x_presets.templates.data_diskcol_profile import neko_diskcol_template
from osocrNG.sptokens import tSPLIT_OCR,tSPLIT_ITEM,tSPLIT_PANOPTIC
from typing import List, Optional, Dict, Any

class neko_augcol_template:
    TYPE_NOAUG="noaug";
    TYPE_ABIAUG="abiaug";

    DSKCOL_TEMPLATE="col_template";
    AUG_PARAM="augparam";
    AUG_TYPE="augtype";
    @classmethod
    def get_default_noaug(cls,coltemplate=None):
        if coltemplate is None:
            coltemplate=neko_diskcol_template.get_default_diskcol_template()
        return {
            cls.DSKCOL_TEMPLATE:coltemplate,
            cls.AUG_TYPE:cls.TYPE_NOAUG,
            cls.AUG_PARAM:{}
        };

    @classmethod
    def get_default_abiaug(cls, coltemplate=None):
        if coltemplate is None:
            coltemplate = neko_diskcol_template.get_default_diskcol_template()
        return {
            cls.DSKCOL_TEMPLATE: coltemplate,
            cls.AUG_TYPE: cls.TYPE_ABIAUG,
            cls.AUG_PARAM: {}
        };

class neko_training_load_balance_profile:
    NAME="name"
    TEMPLATE_NAMES="template_names";
    TEMPLATES="templates";
    TMPL_ALLOWED_AR_RANGE = "training_ar_range"; # balancing long, short and vert at loading time
    BATCH_SIZE="batch_size";
    @classmethod
    def get_default_item_ldr_profile(cls,bsf):
        return {
            cls.NAME: "default_char_ldr_bal",
            cls.TEMPLATE_NAMES: ["char"],
            cls.TEMPLATES:{
                "char":{
                    cls.BATCH_SIZE: int(bsf),
                    cls.TMPL_ALLOWED_AR_RANGE: [-9999,9999]
                }
            }
        };
    @classmethod
    def get_default_hvr_word_ldr_profile(cls,bsf):
        return {
            cls.NAME:"default_hvr",
            cls.TEMPLATE_NAMES:["hori","vert","rect"],
            cls.TEMPLATES:{
                "hori":{
                    cls.BATCH_SIZE: int(bsf),
                    cls.TMPL_ALLOWED_AR_RANGE: [1, 9999]
                },
                "rect": {
                    cls.BATCH_SIZE: int(bsf),
                    cls.TMPL_ALLOWED_AR_RANGE: [0.3,3]
                },
                "vert": {
                    cls.BATCH_SIZE: int(bsf),
                    cls.TMPL_ALLOWED_AR_RANGE: [-1, 1]
                }
            }
        }
class neko_testing_load_balance_profile:
    BATCH_SIZE="batch_size";
    @classmethod
    def get_default_ocr_ldr_profile(cls,bsf):
        return {
            cls.BATCH_SIZE:bsf
        };
class neko_db_profile:
    SRCS = "srcs";
    NAME= "name";
    TYPE="type";
    TOKENIZER_KEY="tokenizer_key";
    FORCE_GRAY="force_gray"

    @classmethod
    def get_default_im_text_lmdb_loader(cls, srcs,name,tokenizer_key,force_grey_scale=False):
        return {
            cls.SRCS: srcs["dict"],
            cls.NAME:name,
            cls.TYPE: "im_text_lmdb",
            cls.FORCE_GRAY: force_grey_scale,
            cls.TOKENIZER_KEY: tokenizer_key
        };

    @classmethod
    def get_panoptic_lmdb_loader(cls, srcs, name, tokenizer_key):
        return {
            cls.SRCS: srcs["dict"],
            cls.NAME: name,
            cls.TYPE: "panoptic_lmdb",
            cls.TOKENIZER_KEY: tokenizer_key
        };


# implication here is we are giving the control for augmentation to individual data groups.
# post alpha2, lscts are managed separately.
# mainly bcs lscts can come with their own meta, so better not mixing them here.
class neko_datastream_profile:
    DB="db";
    NAME="name";
    BALCFG="balcfg";
    AUGCOLS = "augmentation_collations";
    LDRIDX="loader_index";
    SIGNATURES="signatures";
    CONST_LODR_IDX_PATH="loader_index";
    MAXT="maxt"
    ENDPOINTS="endpoints";

    @classmethod
    def get_abstract_training_im_item_loader(cls,dbname,data_root,srcs,data_signature,batchsize,augcols:Dict[str,Any],force_grey_scale=False):
        cfg= {
            cls.DB:neko_db_profile.get_default_im_text_lmdb_loader(srcs,dbname,tSPLIT_ITEM,force_grey_scale=force_grey_scale),
            cls.BALCFG: neko_training_load_balance_profile.get_default_item_ldr_profile(batchsize),
            cls.AUGCOLS: augcols,
            cls.SIGNATURES:{}
        }
        idir = os.path.join(data_root, cls.CONST_LODR_IDX_PATH);
        cfg[cls.NAME]=P(cfg[cls.DB][neko_db_profile.NAME],cfg[cls.BALCFG][neko_training_load_balance_profile.NAME]);
        dep=P(dbname,"-".join(augcols.keys()));
        cfg[cls.SIGNATURES][dep]=data_signature;
        cfg[cls.LDRIDX] = os.path.join(idir, cfg[cls.NAME]+ ".pkl");
        cfg[cls.ENDPOINTS]=[dep];
        cfg[cls.MAXT]=1;

        return cfg;

    @classmethod
    def get_default_training_im_item_loader(cls,dbname,data_root,srcs,data_signature,batchsize,force_grey_scale=False):
        augcols={"noaug": neko_augcol_template.get_default_noaug()};
        return cls.get_abstract_training_im_item_loader(dbname,data_root,srcs,data_signature,batchsize,augcols,force_grey_scale=force_grey_scale)

    @classmethod
    def get_abi_aug_training_im_item_loader(cls,dbname,data_root,srcs,data_signature,batchsize,force_grey_scale=False):
        augcols = {"abiaug": neko_augcol_template.get_default_abiaug()};
        return cls.get_abstract_training_im_item_loader(dbname, data_root, srcs, data_signature, batchsize, augcols,force_grey_scale=force_grey_scale)

    @classmethod
    def get_default_testing_im_item_loader(cls, dbname, srcs,data_signature,batchsize,force_grey_scale=False):
        cfg = {
            cls.DB: neko_db_profile.get_default_im_text_lmdb_loader(srcs,dbname,tSPLIT_ITEM,force_grey_scale=force_grey_scale),
            cls.AUGCOLS: {"noaug": neko_augcol_template.get_default_noaug()}, # testing will never augment
            cls.BALCFG: neko_testing_load_balance_profile.get_default_ocr_ldr_profile(batchsize),
            cls.SIGNATURES: {}
        };
        dep=P(dbname, "noaug");
        cfg[cls.ENDPOINTS] = [dep];
        cfg[cls.SIGNATURES][dep]=data_signature;
        cfg[cls.MAXT] = 1;
        return cfg;

    @classmethod
    def get_default_testing_im_panoptic_loader(cls, dbname, srcs, data_signature, batchsize,force_grey_scale=False):
        cfg = {
            cls.DB: neko_db_profile.get_default_im_text_lmdb_loader(srcs, dbname,tSPLIT_PANOPTIC,force_grey_scale=force_grey_scale),
            cls.AUGCOLS: {"noaug": neko_augcol_template.get_default_noaug()},  # testing will never augment
            cls.BALCFG: neko_testing_load_balance_profile.get_default_ocr_ldr_profile(batchsize),
            cls.SIGNATURES: {}
        };
        dep = P(dbname, "noaug");
        cfg[cls.ENDPOINTS] = [dep];
        cfg[cls.SIGNATURES][dep] = data_signature;
        cfg[cls.MAXT] = 1;
        return cfg;

    @classmethod
    def get_default_training_im_panoptic_loader(cls,dbname,data_root,srcs,data_signature,batchsize,maxT=9999):
        cfg= {
            cls.DB:neko_db_profile.get_panoptic_lmdb_loader(srcs,dbname,tSPLIT_PANOPTIC),
            cls.AUGCOLS: {"noaug": neko_augcol_template.get_default_noaug()},  # testing will never augment
            cls.BALCFG: neko_training_load_balance_profile.get_default_item_ldr_profile(batchsize),
            cls.SIGNATURES: {}
        }
        idir = os.path.join(data_root, cls.CONST_LODR_IDX_PATH);
        dep=P(dbname,"noaug");
        cfg[cls.SIGNATURES][dep]=data_signature;
        cfg[cls.NAME]=P(cfg[cls.DB][neko_db_profile.NAME],cfg[cls.BALCFG][neko_training_load_balance_profile.NAME]);
        cfg[cls.ENDPOINTS] = [P(dbname, "noaug")];
        cfg[cls.LDRIDX] = os.path.join(idir, cfg[cls.NAME]+ ".pkl");# even if you don't need load balance--- it is needed to weed out bad images.
        cfg[cls.MAXT] = maxT;
        return cfg;
    @classmethod
    def get_default_training_im_seq_loader(cls,dbname,data_root,srcs,data_signature,batchsize,maxT=32,force_grey_scale=False):
        cfg= {
            cls.DB:neko_db_profile.get_default_im_text_lmdb_loader(srcs,dbname,tSPLIT_OCR,force_grey_scale=force_grey_scale),
            cls.BALCFG: neko_training_load_balance_profile.get_default_hvr_word_ldr_profile(batchsize),
            cls.AUGCOLS: {"abiaug": neko_augcol_template.get_default_abiaug()},
            cls.SIGNATURES: {}
        }
        idir = os.path.join(data_root, cls.CONST_LODR_IDX_PATH);
        dep=P(dbname,"abiaug");
        cfg[cls.SIGNATURES][dep]=data_signature;
        cfg[cls.NAME]=P(cfg[cls.DB][neko_db_profile.NAME],cfg[cls.BALCFG][neko_training_load_balance_profile.NAME]);
        cfg[cls.LDRIDX] = os.path.join(idir, cfg[cls.NAME]+ ".pkl");
        cfg[cls.ENDPOINTS] = [P(dbname, "abiaug")];
        cfg[cls.MAXT] = maxT;
        return cfg;
    @classmethod
    def get_default_testing_im_seq_loader(cls,dbname,srcs,data_signature,batchsize,maxT=32,force_grey_scale=False):
        cfg = {
            cls.DB: neko_db_profile.get_default_im_text_lmdb_loader(srcs,dbname,tSPLIT_OCR,force_grey_scale=force_grey_scale),
            cls.AUGCOLS: {"noaug": neko_augcol_template.get_default_noaug()}, # testing will never augment
            cls.BALCFG: neko_testing_load_balance_profile.get_default_ocr_ldr_profile(batchsize),
            cls.SIGNATURES: {}
        };
        dep=P(dbname, "noaug");
        cfg[cls.ENDPOINTS] = [dep];
        cfg[cls.SIGNATURES][dep]=data_signature;
        cfg[cls.MAXT] = maxT; # this is a hint to model builder.
        return cfg;