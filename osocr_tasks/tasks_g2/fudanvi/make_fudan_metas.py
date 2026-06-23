import os.path

from osocr_tasks.tasksg1.dscs import makept
from neko_sdk.lmdb_wrappers.splitds import  harvast_cs
from neko_sdk.ocr_modules.charset.fudanvi import fudanch_new
from osocr_tasks.ds_paths import (
    get_fudanvi_web_val,get_fudanvi_web_test,get_fudanvi_web_train,
    get_fudanvi_scene_val,get_fudanvi_scene_test,get_fudanvi_scene_train,
    get_fudanvi_document_val,get_fudanvi_document_train,get_fudanvi_document_test,
    get_fudanvi_hwdb_val, get_fudanvi_hwdb_train, get_fudanvi_hwdb_test,

)

def make_meta(dsgrp,meta_name,fnts):
    tar=set();
    for i in dsgrp:
        ics=harvast_cs(i);
        tar=tar.union(ics);
    makept(None,
           fnts,
           os.path.join("/home/lasercat/ssddata/dicts/",meta_name),tar,set());

ALL={
    "dabfudan_web_trval.pt": [get_fudanvi_web_val,get_fudanvi_web_train],
    "dabfudan_web_test.pt": [get_fudanvi_web_test],
    "dabfudan_scene_trval.pt": [get_fudanvi_scene_val,get_fudanvi_scene_train],
    "dabfudan_scene_test.pt": [get_fudanvi_scene_test],
    "dabfudan_document_trval.pt": [get_fudanvi_document_val, get_fudanvi_document_train],
    "dabfudan_document_test.pt": [get_fudanvi_document_test],
    "dabfudan_hwdb_trval.pt": [get_fudanvi_hwdb_val, get_fudanvi_hwdb_train],
    "dabfudan_hwdb_test.pt": [get_fudanvi_hwdb_test],
    "dabfudanall_trval.pt": [
        get_fudanvi_web_val,get_fudanvi_web_train,
        get_fudanvi_scene_val,get_fudanvi_scene_train,
        get_fudanvi_document_val, get_fudanvi_document_train,
        get_fudanvi_hwdb_val, get_fudanvi_hwdb_train
    ],
    "dabfudanall_test.pt":[
        get_fudanvi_web_test,get_fudanvi_scene_test,get_fudanvi_document_test,get_fudanvi_hwdb_test
    ]
}

if __name__ == '__main__':
    fnts=["/home/lasercat/ssddata/metamk/NotoSansCJKsc-Regular.otf"];

    # for meta_name in ALL:
    #     make_meta([f() for f in ALL[meta_name]],meta_name,fnts);
    makept(None,
           fnts,
           os.path.join("/home/lasercat/ssddata/dicts/", "dabfudangithub"), fudanch_new,
           set());


