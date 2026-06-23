import os
from osocr_tasks.ds_paths import (get_istr_bengali_test, get_istr_hindi_test, get_istr_kannada_train,
                                  get_istr_malayalam_test,
                                  get_istr_marathi_train, get_istr_punjabi_test, get_istr_tamil_train,
                                  get_istr_telugu_test,
                                  get_istr_gujarati_test, get_istr_hindi_train, get_istr_kannada_test,
                                  get_istr_malayalam_train, get_istr_marathi_test, get_istr_punjabi_train,
                                  get_istr_tamil_test,
                                  get_istr_telugu_train, get_istr_odia_train, get_istr_odia_test, get_istr_odia_val,
                                  get_istr_bengali_train, get_istr_gujarati_train,
                                  get_mj_tr_abi,get_st_abi,get_iiit5k
                                  )
# training data profiles.
class neko_data_repo_cfg:
    PARAM_name="name";
    PARAM_root_dict="dict";

def get_chslat_v1(dataroot):
     return{
         "name": "chslat_v1",
         "dict":{
                "art":  os.path.join(dataroot,'artdb_seen_NG'),
                "mlt":  os.path.join(dataroot,'mlttrchlat_seen_NG'),
                "ctw":  os.path.join(dataroot,'ctwdb_seen_NG'),
                "rctw":  os.path.join(dataroot,'rctwtrdb_seen_NG'),
                "lsvt":  os.path.join(dataroot,'lsvtdb_seen_NG')
            }
     };

# cross task sharing, if you need to.
def get_istr_v1(dataroot):
    return {
        "name": "istr_v1",
        "dict": {
            "bengali": get_istr_bengali_train(dataroot),
            "gujarati": get_istr_gujarati_train(dataroot),
            "hindi": get_istr_hindi_train(dataroot),
            "kannada": get_istr_kannada_train(dataroot),
             "malayalam": get_istr_malayalam_train(dataroot),
             "marathi": get_istr_marathi_train(dataroot),
             "telugu": get_istr_telugu_train(dataroot),
             "odia": get_istr_odia_train(dataroot),
        }
    };
# cross task sharing, if you need to.
def get_abi_mjst(dataroot):
    return {
        "name": "abimjst",
        "dict": {
            "MJ": get_mj_tr_abi(dataroot),
            "ST":get_st_abi(dataroot)
        }
    };
# cross task sharing, if you need to.
def get_ctw_yolox_tr(dataroot):
    return {
        "typename": "ctw_ch_tr",
        "dict": {
            "ctw_ch": os.path.join(dataroot,"ctw_ch_yolox/train/"),
        }
    };
# cross task sharing, if you need to.
def get_ctw_yolox_te(dataroot):
    return {
        "typename": "ctw_ch_te",
        "dict": {
            "ctw_ch": os.path.join(dataroot,"ctw_ch_yolox/test/"),
        }
    };

def get_enword_te(dataroot):
    return {
        "typename": "enword_te",
        "dict": {
            "iiit5k": get_iiit5k(dataroot),
        }
    };
def get_ctw_yolox_val(dataroot):
    return {
        "typename": "ctw_ch_val",
        "dict": {
            "ctw_ch": os.path.join(dataroot,"ctw_ch_yolox/val/"),
        }
    };
