import os.path

from osocr_tasks.tasksg1.dscs import makept_core
from neko_sdk.lmdb_wrappers.splitds import  harvast_cs
from neko_sdk.ocr_modules.charset.fudanvi import fudanch_new
from osocr_tasks.ds_paths import get_indicstr12_synth,INDIC_STR12
from osocr_tasks.ds_paths import get_hhd_train,get_hhd_test,get_hhd_test_1800ad
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.symbols import symbol

ROOT = "/run/media/lasercat/portable/ssddata/indicstr12synth/";

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



# def add_3755(tar,fnts,allc,allfid,renderwith="/home/lasercat/ssddata/metamk/NotoSansCJKsc-Regular.otf"):
#     fnts.append(renderwith);
#     fid=len(fnts)-1;
#     for i in t1_3755:
#         if i not in tar:
#             tar[i]=[fid];
#             allc.append(i);
#             allfid.append(fid);
#     return tar, fnts, allc, allfid;
def write_meta(meta_name, tar, fnts, allc, allfid):
    folder=os.path.join("/home/lasercat/ssddata/dictsv2/",meta_name);
    os.makedirs(folder,exist_ok=True);
    makept_core(allc,
           fnts,allfid,
           os.path.join(folder,"vismeta.pt"),tar,set());


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
    # for meta_name in ALL:
    #     make_meta([f for f in ALL[meta_name]],meta_name);
    # make_meta(["hindi","kannada","malayalam","marathi","punjabi","tamil","telugu"],"dab_moostrHI",[t1_3755],[""])
    tar = {};
    fnts = [];
    allc = [];
    allfid = [];
    # tar, fnts, allc, allfid = add_from_lmdb(["hindi","kannada","malayalam","marathi","punjabi","tamil","telugu"], tar, fnts, allc, allfid,ROOT);

    tar, fnts, allc, allfid = add_from_lmdb(["ethiopic"],
                                            tar, fnts, allc, allfid, "/home/lasercat/berzelius/shared/data_mose/ssddata/hhd/");

    # tar, fnts, allc, allfid = add_latin_62(tar, fnts, allc, allfid);
    # tar,fnts,allc,allfid=add_3755(tar,fnts,allc,allfid);
    # tar, fnts, allc, allfid = clean_symbols(tar, fnts, allc, allfid);
    # write_meta("dab_ethopic", tar, fnts, allc, allfid);
