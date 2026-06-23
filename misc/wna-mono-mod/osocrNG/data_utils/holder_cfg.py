import os.path

from osocr_tasks.ds_paths import get_cvpr16, get_nips14
from osocr_tasks.ds_paths import (
    get_fudanvi_document_train, get_fudanvi_document_val,get_fudanvi_document_test,
    get_fudanvi_web_train,get_fudanvi_web_val,get_fudanvi_web_test,
    get_fudanvi_hwdb_train,get_fudanvi_hwdb_val,get_fudanvi_hwdb_test,
    get_fudanvi_scene_train,get_fudanvi_scene_val,get_fudanvi_scene_test,
    get_mj_val_abi,get_st_abi,get_mj_tr_abi,get_mj_te_abi)


from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
def get_cvpr_2016_holder(dataroot):
    return neko_lmdb_holder({"root": get_cvpr16(dataroot)});


def get_nips_2014_holder(dataroot):
    return neko_lmdb_holder({"root": get_nips14(dataroot)});

def get_fudanvi_scene_tr_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_scene_train(dataroot)});

def get_fudanvi_web_tr_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_web_train(dataroot)});
def get_fudanvi_hwdb_tr_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_hwdb_train(dataroot)});
def get_fudanvi_doc_tr_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_document_train(dataroot)});

def get_fudanvi_scene_te_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_scene_test(dataroot)});

def get_fudanvi_web_te_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_web_test(dataroot)});
def get_fudanvi_hwdb_te_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_hwdb_test(dataroot)});
def get_fudanvi_doc_te_holder(dataroot):
    return neko_lmdb_holder({"root":get_fudanvi_document_test(dataroot)});



def get_mjst_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dab62cased");

def get_fudanvi_web_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_web_trval");
def get_fudanvi_hwdb_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_hwdb_trval");
def get_fudanvi_doc_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_document_trval");
def get_fudanvi_scene_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_scene_trval");

def get_fudanvi_all_tr_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudanall_trval");

def get_fudanvi_web_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_web_test");
def get_fudanvi_hwdb_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_hwdb_test");
def get_fudanvi_doc_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_document_test");
def get_fudanvi_scene_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_scene_test");

def get_fudanvi_all_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudanall_test");
def get_fudan_github_test_meta(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudangithub");


def get_mjst_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dab62cased");

def get_fudanvi_web_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_web_trval");
def get_fudanvi_hwdb_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_hwdb_trval");
def get_fudanvi_doc_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_document_trval");
def get_fudanvi_scene_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_scene_trval");

def get_fudanvi_all_tr_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudanall_trval");

def get_fudanvi_web_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_web_test");
def get_fudanvi_hwdb_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_hwdb_test");
def get_fudanvi_doc_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_document_test");
def get_fudanvi_scene_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudan_scene_test");

def get_fudanvi_all_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudanall_test");
def get_fudan_github_test_metav2(dsroot):
    return os.path.join(dsroot, "dictsv2", "dabfudangithub");


def get_abi_mj_tr_holder(dataroot):
    return neko_lmdb_holder({"root": get_mj_tr_abi(dataroot)});

def get_abi_mj_val_holder(dataroot):
    return neko_lmdb_holder({"root": get_mj_val_abi(dataroot)});

def get_abi_mj_te_holder(dataroot):
    return neko_lmdb_holder({"root": get_mj_te_abi(dataroot)});

def get_abi_st_holder(dataroot):
    return neko_lmdb_holder({"root": get_st_abi(dataroot)});

