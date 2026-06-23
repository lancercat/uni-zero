import os.path

from osocr_tasks.tasksg1.dscs import makept_core
from neko_sdk.lmdb_wrappers.splitds import  harvast_cs
from neko_sdk.ocr_modules.charset.fudanvi import fudanch_new
from osocr_tasks.ds_paths import (
    get_istr_bengali_test,get_istr_bengali_val,get_istr_gujarati_train,get_istr_hindi_test,
    get_istr_hindi_val,get_istr_kannada_train,get_istr_malayalam_test,get_istr_malayalam_val,
    get_istr_marathi_train,get_istr_punjabi_test,get_istr_punjabi_val,get_istr_tamil_train,
    get_istr_telugu_test,get_istr_telugu_val,get_istr_bengali_train,get_istr_gujarati_test,
    get_istr_gujarati_val,get_istr_hindi_train,get_istr_kannada_test,get_istr_kannada_val,
    get_istr_malayalam_train,get_istr_marathi_test,get_istr_marathi_val,get_istr_punjabi_train,
    get_istr_tamil_test,get_istr_tamil_val,get_istr_telugu_train,
get_istr_odia_test,get_istr_odia_val,get_istr_odia_train)
from osocr_tasks.ds_paths import get_hhd_train,get_hhd_test,get_hhd_test_1800ad
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.symbols import symbol

ROOT = "/run/media/lasercat/portable/ssddata/istr/";

GRPS={
"bengali": [get_istr_bengali_test,get_istr_bengali_val,get_istr_bengali_train],
"gujarati":[get_istr_gujarati_test,get_istr_gujarati_val,get_istr_gujarati_train],
"hindi":[get_istr_hindi_test,get_istr_hindi_val,get_istr_hindi_train],
"kannada":[  get_istr_kannada_train, get_istr_kannada_test,get_istr_kannada_val],
"malayalam":[get_istr_malayalam_test,get_istr_malayalam_val,get_istr_malayalam_train],
"marathi":[get_istr_marathi_train,get_istr_marathi_test,get_istr_marathi_val],
"punjabi":[get_istr_punjabi_test,get_istr_punjabi_val,get_istr_punjabi_train],
"tamil":[get_istr_tamil_train,get_istr_tamil_test,get_istr_tamil_val],
"telugu":[get_istr_telugu_test,get_istr_telugu_val,get_istr_telugu_train],
"odia": [get_istr_odia_test,get_istr_odia_val,get_istr_odia_train],
"ethiopic": [get_hhd_train, get_hhd_test, get_hhd_test_1800ad]

}

ALL={

    "dab_istr_bengali": ["bengali"],
    "dab_istr_gujarati": ["gujarati"],
    "dab_istr_hindi": ["hindi"],
    "dab_istr_odia": ["odia"],
    "dab_istr_kannada": ["kannada"],
    "dab_istr_malayalam": ["malayalam"],
    "dab_istr_marathi": ["marathi"],
    "dab_istr_punjabi": ["punjabi"],
    "dab_istr_tamil": ["tamil"],
    "dab_istr_telugu": ["telugu"],
    "dab_istr_notbg": ["bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi", "telugu", "odia"],
    "dab_istr_bg": ["bengali", "gujarati"]
    };
def add_from_lmdb(langgrp,tar,fnts,allc,allfid,root):
    tar_loc = set();
    for i in langgrp:
        fnts.append(os.path.join(root, i + ".ttf"));
        for d in GRPS[i]:
            ics = harvast_cs(d());
            for c in ics:
                if(c not in tar):
                    tar[c] = len(fnts) - 1;
                    tar_loc.add(c);
    for c in tar_loc:
        allc.append(c);
        allfid.append(tar[c]);
    return tar,fnts,allc,allfid;
def add_latin_62(tar,fnts,allc,allfid):
    for i in latin62:
        if i not in tar:
            tar[i]=0;
            allc.append(i);
            allfid.append(0);  # everyone renders english
    return tar,fnts,allc,allfid;



def add_3755(tar,fnts,allc,allfid,renderwith="/home/lasercat/ssddata/metamk/NotoSansCJKsc-Regular.otf"):
    fnts.append(renderwith);
    fid=len(fnts)-1;
    for i in t1_3755:
        if i not in tar:
            tar[i]=[fid];
            allc.append(i);
            allfid.append(fid);
    return tar, fnts, allc, allfid;
def write_meta(meta_name, tar, fnts, allc, allfid):
    folder=os.path.join("/home/lasercat/ssddata/dictsv2/",meta_name);
    os.makedirs(folder,exist_ok=True);
    fn=os.path.join(folder,"vismeta.pt");
    makept_core(allc,
           fnts,allfid,
           fn,tar,set());
    print("saved to", fn)


def make_meta(langgrp,meta_name):
    tar={};
    fnts=[];
    allc=[];
    allfid=[];
    tar,fnts,allc,allfid=add_from_lmdb(langgrp,tar,fnts,allc,allfid,ROOT);
    tar, fnts, allc, allfid = add_latin_62(tar, fnts, allc, allfid);
    write_meta(meta_name,tar,fnts,allc,allfid);


def clean_symbols(tar, fnts, allc, allfid):
    ntar = {};
    nallc = [];
    nallfid = [];
    for cid in range(len(allc)):
        c=allc[cid];
        if c not in symbol:
            ntar[c]=tar[c];
            nallc.append(c);
            nallfid.append(allfid[cid]);
        else:
            print("removing symbol",c);
    return ntar, fnts, nallc, nallfid;


if __name__ == '__main__':
    # make_meta([f for f in ALL["dab_istr_bg"]],"dab_istr_bg");

    for meta_name in ALL:
        make_meta([f for f in ALL[meta_name]],meta_name);
    # make_meta(["hindi","kannada","malayalam","marathi","punjabi","tamil","telugu"],"dab_moostrHI",[t1_3755],[""])
    # tar = {};
    # fnts = [];
    # allc = [];
    # allfid = [];
    # tar, fnts, allc, allfid = add_from_lmdb(["hindi","kannada","malayalam","marathi","punjabi","tamil","telugu"], tar, fnts, allc, allfid,ROOT);
    #
    # tar, fnts, allc, allfid = add_from_lmdb(["ethiopic"],
    #                                         tar, fnts, allc, allfid, "/home/lasercat/berzelius/shared/data_mose/ssddata/hhd/");

    # tar, fnts, allc, allfid = add_latin_62(tar, fnts, allc, allfid);
    # tar,fnts,allc,allfid=add_3755(tar,fnts,allc,allfid);
    # tar, fnts, allc, allfid = clean_symbols(tar, fnts, allc, allfid);
    # write_meta("dab_ethopic", tar, fnts, allc, allfid);
