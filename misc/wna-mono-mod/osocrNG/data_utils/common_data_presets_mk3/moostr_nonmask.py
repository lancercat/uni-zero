
import os.path
import time

import cv2
import tqdm

from osocrNG.data_utils.common_data_presets.dspathsNG import get_mltkr_path,get_mltjphv_path,get_mltjp_path
from osocrNG.data_utils.neko_imageset_holder import neko_image_holder
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.athena.common.analyze_folder import bootstrap_folder
from osocrNG.data_utils.common_data_presets_mk3.factory import abstract_mk3_data_factory
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent,augment_agent_abinet,augment_agent_abinet_zeropadding,augment_agent_abinet_fixed

def get_chs_training_metav2(root):
    return os.path.join(root, "dictsv2", "dab3791MC");
def get_synth_rand_3755_8_1(root):
    return os.path.join(root,"random-3k-1-8chs3755-1-lmdb/");
def get_synth_rand_3755_8_heavy(root):
    return os.path.join(root,"random-30k-1-8chs3755-1-lmdb/");

def get_synth_rand_3755_8_tpssspfailure(root):
    return os.path.join(root,"random-3k-1-8tpsSSP_bad_chars_3755-1-lmdb/");
def get_synth_rand_kr_8_tpssspfailure(root):
    return os.path.join(root,"random-3k-1-8tpsSSP_bad_chars_kr-1-lmdb/");
def get_synth_rand_uk_8_tpssspfailure(root):
    return os.path.join(root,"random-3k-1-8tpsSSP_bad_chars_uk-1-lmdb/");

def get_synth_rand_3755_8_2(root):
    return  os.path.join(root,"random-3k-1-8chs3755-2-lmdb/");
def get_synth_rand_ukanji_8_1(root):
    return  os.path.join(root,"random-3k-1-8ukanji-1-lmdb/");
def get_synth_rand_ukanji_8_1_heavy(root):
    return  os.path.join(root,"random-30k-1-8ukanji-1-lmdb/");
def get_synth_rand_ukanji_8_2(root):
    return  os.path.join(root,"random-3k-1-8ukanji-2-lmdb/");

def get_synth_rand_hirakata_8_1(root):
    return  os.path.join(root,"random-3k-1-8hirakata-1-lmdb/");
def get_synth_rand_hirakata_8_2(root):
    return  os.path.join(root,"random-3k-1-8hirakata-2-lmdb/");

def get_synth_rand_kr_8_1(root):
    return  os.path.join(root,"random-3k-1-8kr-1-lmdb/");
def get_synth_rand_kr_8_heavy(root):
    return  os.path.join(root,"random-30k-1-8kr-1-lmdb/");
def get_synth_rand_kr_8_2(root):
    return  os.path.join(root,"random-3k-1-8kr-2-lmdb/");
def get_synth_rand_3755_unary_1(root):
    return  os.path.join(root,"random-3kchs3755-unary_1-lmdb/");
def get_synth_rand_3755_unary_2(root):
    return  os.path.join(root,"random-3kchs3755-unary_2-lmdb/");
def get_synth_rand_ukanji_unary_1(root):
    return  os.path.join(root,"random-3kukanji-unary_1-lmdb/");
def get_synth_rand_ukanji_unary_2(root):
    return  os.path.join(root,"random-3kukanji-unary_2-lmdb/");

def get_synth_rand_hirakata_unary_1(root):
    return  os.path.join(root,"random-3khirakata-unary_1-lmdb/");
def get_synth_rand_hirakata_unary_2(root):
    return  os.path.join(root,"random-3khirakata-unary_2-lmdb/");

def get_synth_rand_kr_unary_1(root):
    return  os.path.join(root,"random-3kkr-unary_1-lmdb/");
def get_synth_rand_kr_unary_2(root):
    return  os.path.join(root,"random-3k8kr-unary_2-lmdb/");
def get_synth_rand_3755_full_ctx_1(root):
    return  os.path.join(root,"random-3kchs3755-full_ctx_1-lmdb/");
def get_synth_rand_3755_full_ctx_2(root):
    return  os.path.join(root,"random-3kchs3755-full_ctx_2-lmdb/");
def get_synth_rand_ukanji_full_ctx_1(root):
    return  os.path.join(root,"random-3kukanji-full_ctx_1-lmdb/");
def get_synth_rand_ukanji_full_ctx_2(root):
    return  os.path.join(root,"random-3kukanji-full_ctx_2-lmdb/");

def get_synth_rand_hirakata_full_ctx_1(root):
    return  os.path.join(root,"random-3khirakata-full_ctx_1-lmdb/");
def get_synth_rand_hirakata_full_ctx_2(root):
    return  os.path.join(root,"random-3khirakata-full_ctx_2-lmdb/");

def get_synth_rand_kr_full_ctx_1(root):
    return  os.path.join(root,"random-3kkr-full_ctx_1-lmdb/");
def get_synth_rand_kr_full_ctx_2(root):
    return  os.path.join(root,"random-3k8kr-full_ctx_2-lmdb/");
# using factory class to avoid wild strings.
# you can still use wild strings to call the factory,
# but now you can not to.
class moostr_mk3_data_factory(abstract_mk3_data_factory):
    @ classmethod
    def arm_osocr_test_jpn_full(cls,meta_dict,data_dict,test_dict,dataroot,v2h=-9):
        FNS = [get_mltjp_path];
        NMS = ["JPN"];
        metadict_ = {
            "JPN-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlt"),
                 "case_sensitive": False,
                 "has_unk": False,
                 },
            "JPN-OSR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_osr"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPN-GOSR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPN-OSTR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_kanji"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];
        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict,data_dict,test_dict

    @classmethod
    def arm_osocr_test_jpnhv_gosr(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GOSR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_osocr_test_jpnhv_full(cls,meta_dict,data_dict,test_dict,dataroot,v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 },
            "JPNHV-OSR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_osr"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPNHV-GOSR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPNHV-OSTR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_kanji"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not  in meta_dict);
            meta_dict[m]=metadict_[m];
        data_dict_={}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n]=data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict,data_dict,test_dict

    @classmethod
    def arm_osocr_test_jpnhv_gzsl(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_osocr_test_kr_full(cls,meta_dict,data_dict,test_dict,dataroot,v2h=-9):
        FNS = [get_mltkr_path];
        NMS = ["KR"];
        metadict_ = {
            "KR-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk":False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not  in meta_dict);
            meta_dict[m]=metadict_[m];
        data_dict_={}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n]=data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict,data_dict,test_dict

    @classmethod
    def arm_synth_test_3755(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_3755_8_1,get_synth_rand_3755_8_2,
               get_synth_rand_3755_unary_1, get_synth_rand_3755_unary_2,
               get_synth_rand_3755_full_ctx_1, get_synth_rand_3755_full_ctx_2
               ];
        NMS = ["RAND_8_1","RAND_8_2",
               "RAND_U_1","RAND_U_2",
               "RAND_F_1","RAND_F_2"

               ];
        metadict_ = {
            "SL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dab3791MC"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_3755_adhoc(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_3755_8_tpssspfailure];
        NMS = ["RAND_8_1_tpsSSPfail_SL"];
        metadict_ = {
            "SL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dab3791MC"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_synth_test_3755_heavy(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_3755_8_heavy];
        NMS = ["RAND_8_1"];
        metadict_ = {
            "SL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dab3791MC"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict

    @classmethod
    def arm_synth_test_jpn_heavy(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_ukanji_8_1_heavy];
        NMS = ["UKANJI_8_1"];
        metadict_ = {
            "JPN":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_jpn_adhoc(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_uk_8_tpssspfailure];
        NMS = ["RAND_8_1_tpsSSPfail_JPN"];
        metadict_ = {
            "JPN":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_jpn(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_ukanji_8_1,get_synth_rand_ukanji_8_2,get_synth_rand_hirakata_8_1,get_synth_rand_hirakata_8_2,
                  get_synth_rand_ukanji_unary_1,get_synth_rand_ukanji_unary_2,get_synth_rand_hirakata_unary_1,get_synth_rand_hirakata_unary_2,
               get_synth_rand_ukanji_full_ctx_1, get_synth_rand_ukanji_full_ctx_2, get_synth_rand_hirakata_full_ctx_1,get_synth_rand_hirakata_full_ctx_2,
               ];
        NMS = [
                  "UKANJI_8_1","UKANJI_8_2","HIRAKATA_8_1","HIRAKATA_8_2",
                  "UKANJI_U_1","UKANJI_U_2","HIRAKATA_U_1","HIRAKATA_U_2",
                  "UKANJI_F_1", "UKANJI_F_2", "HIRAKATA_F_1", "HIRAKATA_F_2"
               ];
        metadict_ = {
            "JPN":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_kr_heavy(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_kr_8_heavy];
        NMS = ["KR_8_1"];
        metadict_ = {
            "KR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_kr_adhoc(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_kr_8_tpssspfailure];
        NMS = ["RAND_8_1_tpsSSPfail_KR"];
        metadict_ = {
            "KR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def arm_synth_test_kr(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_synth_rand_kr_8_1,get_synth_rand_kr_8_2];
        NMS = ["KR_8_1","KR_8_2"];
        metadict_ = {
            "KR":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
    @classmethod
    def get_osocr_test_gosr(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=cls.arm_osocr_test_jpnhv_gosr(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_all(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=cls.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict,dataroot,v2h);

        meta_dict, data_dict, test_dict=cls.arm_osocr_test_jpnhv_full(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_osocr_test_jpn_full(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_310(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_jpnhv_gzsl(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_legacy(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_jpn_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_heavy_synth(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict=cls.arm_synth_test_kr_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_synth_test_3755_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_synth_test_jpn_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_mk3_test_adhoc(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict=cls.arm_synth_test_3755_adhoc(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_synth_test_kr_adhoc(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict = cls.arm_synth_test_jpn_adhoc(meta_dict, data_dict, test_dict, dataroot, v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };

    @classmethod
    def get_osocr_test_allsync(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=cls.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=cls.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_jpn_hv(cls,dataroot, v2h=-9,pfx=""):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 }
        };
        datadict = {
        };
        for d, n in zip(FNS, NMS):
            datadict[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})

        test_dict = {};
        for d in datadict:
            for m in metadict:
                test_dict[d + "-gzsl"+pfx] = {
                    "data": d,
                    "meta": m,
                }
        return {
            "meta": metadict,
            "data": datadict,
            "tests": test_dict
        };
    @classmethod
    def get_moostr_v1(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["art", "mlt", "ctw", "rctw", "lsvt"],
                he.PARAM_sourced: {
                    "art": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen_NG'), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"moostr-");
    @classmethod
    def get_moostr_v1_mk3(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["art", "mlt", "ctw", "rctw", "lsvt"],
                he.PARAM_sourced: {
                    "art": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen_NG'), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"ostr-");
    @classmethod
    def get_ostr_v1_mk3(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["art", "mlt", "ctw", "rctw", "lsvt"],
                he.PARAM_sourced: {
                    "art": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen'), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"ostr-");
    @classmethod
    def arm_moostr_v1(cls,agent_dict, qdict, params):
        da = cls.get_moostr_v1(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h,params,-9)
        );
        agent_dict,qdict=cls.arm_training_data(da,agent_dict,qdict,params);
        meta=get_chs_training_metav2(params[cls.PARAM_dataroot]);
        return agent_dict,qdict,meta;

    @classmethod
    def arm_moostr_v1_mk3(cls, agent_dict, qdict, params):
        da = cls.get_moostr_v1_mk3(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h, params, -9)
        );
        agent_dict, qdict = cls.arm_training_data(da, agent_dict, qdict, params);
        meta = get_chs_training_metav2(params[cls.PARAM_dataroot]);
        return agent_dict, qdict, meta;
    @classmethod
    def arm_ostr_v1_mk3(cls, agent_dict, qdict, params):
        da = cls.get_ostr_v1_mk3(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h, params, -9)
        );
        agent_dict, qdict = cls.arm_training_data(da, agent_dict, qdict, params);
        meta = get_chs_training_metav2(params[cls.PARAM_dataroot]);
        return agent_dict, qdict, meta;
    @classmethod
    def get_benchmark(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_plus(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_plus_hori_only(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_ostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_all(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_release(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_heavy_synth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_heavy_synth(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_adhoc(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_mk3_test_adhoc(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_testall(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_all(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_testsynth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_allsync(data_root,-9);
        return trad,trqd,trm,tedd;

    @classmethod
    def get_mk3_benchmark_testgosr(cls, data_root, anchor_dict, queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                                {
                                                    cls.PARAM_dataroot: data_root,
                                                    cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                    cls.EXPORT_data_queue_name: queue_name,
                                                    cls.PARAM_anchor_dict: anchor_dict,
                                                }
                                                );
        tedd = cls.get_osocr_test_gosr(data_root, -9);
        return trad, trqd, trm, tedd;
    # @classmethod
    # def get_mk3_benchmark_w_moe(cls,data_root,anchor_dict,queue_name):
    #     trad, trqd = {}, {};
    #
    #     trad, trqd, trm = cls.arm_moostr_v1_mk2(trad, trqd,
    #                                           {
    #                                               cls.PARAM_dataroot: data_root,
    #                                               cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
    #                                               cls.EXPORT_data_queue_name: queue_name,
    #                                               cls.PARAM_anchor_dict: anchor_dict,
    #                                           }
    #                                           );
    #     tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
    #     teddM=cls.get_osocr_test_jpn_hv(data_root,-9,"-MoE");
    #
    #     return trad,trqd,trm,tedd,teddM;


class moostr_mk3_data_factoryAA(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet;

class moostr_mk3_data_factoryAAZ(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_zeropadding;

class moostr_mk3_data_factoryAAF(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_fixed;


# To make the process determinstic, please use only one loader to populate on queue





def get_osocr_test_jpn_hv_full(dataroot,v2h=-9):
    FNS = [get_mltjphv_path];
    NMS = ["JPNHV"];
    metadict = {
        "JPNHV-GZSL":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
             "case_sensitive": False,
             "has_unk": False
             # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
             # For these test dss we need the model to pretend not seeing these characters
             },
        "JPNHV-OSR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_osr"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-GOSR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_nohirakata"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-OSTR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_kanjix"),
             "case_sensitive": False,
             "has_unk": True,
            }
    };
    datadict = {
    };
    for d, n in zip(FNS, NMS):
        datadict[n] = neko_lmdb_holder({"root": d(dataroot),"vert_to_hori":v2h})

    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d + "-"+m] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    };


def get_osocr_test_image_based(data_root,dst,lang,v2h=-9):
    files, ptfile, sfolder, dfolder=bootstrap_folder(data_root,dst,lang,"*.*");
    metadict = {
        "generic":
            {"meta_path": ptfile,
             "case_sensitive": True,
             "has_unk": False
             }
    };
    datadict={
        "generic":neko_image_holder({"files":files,"vert_to_hori":v2h})
    }
    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    },dfolder;


if __name__ == '__main__':
    from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
    from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent

    ad,qd={},{};
    from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1v_6_05 as two_hori_main
    anchors=two_hori_main();

    de=moostr_mk3_data_factoryAA;
    ad,qd,meta=de.arm_moostr_v1(ad,qd,
                                               {
                                                   de.PARAM_dataroot:"/home/lasercat/ssddata",
                                                   de.PARAM_preaug_data_queue_name:"preaug_q",
                                                   de.EXPORT_data_queue_name: "dq",
                                                   de.PARAM_anchor_dict:anchors,
                                               }
                                               );
    from multiprocessing import Queue as mpQueue
    for qk in qd:
        qd[qk]=mpQueue(maxsize=9);
    e=neko_environment(queue_dict=qd);
    for ak in ad:
        ad[ak]["agent"].start(ad[ak]["params"],e,mode="forkserver");
    st=time.time();
    for i in tqdm.tqdm(range(100)):
        aug_data=qd["dq"].get();
        cv2.imshow("meow", aug_data["image"][0]);
        cv2.imshow("meow2", aug_data["image"][-1]);
        cv2.waitKey(0);

    et=time.time();
    for ak in ad:
        ad[ak]["agent"].stop();

    print(et-st);

    pass;
