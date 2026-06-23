import glob
import json
import os
import shutil

import cv2
import numpy as np
import tqdm

from appli.chardet.tdict import tdict
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
import torch
from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
from neko_sdk.neko_framework_NG.libmeta.file_names import FN
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
from neko_sdk.ocr_modules.charset.etc_cset import latin62;
from neko_sdk.neko_framework_NG.libmeta.agents.utils import mk_meta,sorted_utf
from neko_sdk.ocr_modules.renderlite.lib_render import render_lite
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
from neko_sdk.ocr_modules.fontkit.font_utilNG import grab_char_from_fnt
from neko_sdk.log import warn,fatal



def mkmasterdict(masterlst=tokenize("QWERTYUIOPASDFGHJKLZXCVBNM"),servantlst=tokenize("qwertyuiopasdfghjklzxcvbnm")):
    mdict={}
    for m,s in zip(masterlst,servantlst):
        mdict[s]=m;
    return mdict
def mkid_meta(tdict):
    sutf=sorted_utf(tdict);
    idd = {}
    for i, utf in enumerate(sutf):
        assert utf not in idd;
        idd[utf] = i;
    return idd;
def mkglyph_meta(tdict,font):
    sutf = sorted_utf(tdict);
    gd = {}
    engine = render_lite(os=84, fos=32);
    for i, utf in tqdm.tqdm(list(enumerate(sutf))):
        assert utf not in gd;
        gd[utf]=engine.center_draw(64, utf, font);
    return gd;
def mkglyph_meta_multiple_fnts(tdict,font_dir):
    fs=glob.glob(os.path.join(font_dir,"*.ttf"));
    cfdict={};
    for f in fs:
        c,s=grab_char_from_fnt(f);
        for ch in s.union(c):
            for chp in ch:
                if(chp not in cfdict):
                    cfdict[chp]=set();
                cfdict[chp].add(f);

    sutf = sorted_utf(tdict);
    gd = {}
    engine = render_lite(os=84, fos=32);
    for i, utf in tqdm.tqdm(list(enumerate(sutf))):
        assert utf not in gd;
        fnts=set(fs);
        for c in utf:
            if(c not in cfdict):
                warn("unsupported",c);
                continue;
            fnts=fnts.intersection(cfdict[c]);
        sfs=list(fnts);
        gd[utf]=engine.center_draw(64, utf,sfs[0]);
    return gd;
def v3_dict_from_list(utflst,name,root="/home/lasercat/ssddata/dicts_v3/",master_dict=None):
    if master_dict is None:
        master_dict=mkmasterdict();
    metapath=os.path.join(root,name)
    tdictpath = os.path.join(metapath, FN.TDICT);
    shutil.rmtree(metapath,ignore_errors=True);
    os.makedirs(metapath,exist_ok=False);

    tdict = mk_meta(utflst, master_dict);
    torch.save(tdict, tdictpath);

    idpath = os.path.join(metapath, FN.ID_META);
    idd = mkid_meta(tdict);
    torch.save(idd, idpath);

    return tdict;


def v3_dict_from_list_and_fnt(utflst,fnt,name,root="/home/lasercat/ssddata/dicts_v3/",master_dict=None):
    tdict=v3_dict_from_list(utflst, name, root, master_dict);
    metapath=os.path.join(root,name);
    gmpath = os.path.join(metapath, FN.GLYPH_META);
    gd = mkglyph_meta(tdict, fnt);
    torch.save(gd, gmpath);
def from_list_multifnt(utflst,fntdir,name,root="/home/lasercat/ssddata/dicts_v3/",master_dict=None):
    if master_dict is None:
        master_dict=mkmasterdict();
    metapath=os.path.join(root,name)
    tdictpath = os.path.join(metapath, FN.TDICT);
    shutil.rmtree(metapath,ignore_errors=True);
    os.makedirs(metapath,exist_ok=False);

    tdict = mk_meta(utflst, master_dict);
    torch.save(tdict, tdictpath);

    idpath = os.path.join(metapath, FN.ID_META);
    idd = mkid_meta(tdict);
    torch.save(idd, idpath);

    gmpath = os.path.join(metapath, FN.GLYPH_META);
    gd = mkglyph_meta_multiple_fnts(tdict, fntdir);
    torch.save(gd, gmpath);
def from_v2(v2root,fnt,name,root="/home/lasercat/ssddata/dicts_v3/"):
    v2=torch.load(os.path.join(v2root,name,"vismeta.pt"));
    utf_list=v2["chars"];
    mstrd={};
    mstr_lst=[];
    svrnt_lst=[];
    for c in utf_list:
        cid=v2["label_dict"][c];
        mid=v2["master"][cid];
        if(mid==cid):
            assert (mid not in mstrd);
            mstrd[mid]=c;
    for c in utf_list:
        cid=v2["label_dict"][c];
        mid=v2["master"][cid];
        if(mid!=cid):
            mstr_lst.append(mstrd[mid])
            svrnt_lst.append(c);
    md=mkmasterdict(mstr_lst,svrnt_lst);
    return v3_dict_from_list_and_fnt(utf_list,fnt, name, root,master_dict=md);


    pass;

def from_v2_multifnt(v2root,fnt_dir,name,root="/home/lasercat/ssddata/dicts_v3/"):
    v2=torch.load(os.path.join(v2root,name,"vismeta.pt"));
    utf_list=v2["chars"];
    mstrd={};
    mstr_lst=[];
    svrnt_lst=[];
    for c in utf_list:
        cid=v2["label_dict"][c];
        mid=v2["master"][cid];
        if(mid==cid):
            assert (mid not in mstrd);
            mstrd[mid]=c;
    for c in utf_list:
        cid=v2["label_dict"][c];
        mid=v2["master"][cid];
        if(mid!=cid):
            mstr_lst.append(mstrd[mid])
            svrnt_lst.append(c);
    md=mkmasterdict(mstr_lst,svrnt_lst);
    return from_list_multifnt(utf_list,fnt_dir, name, root,master_dict=md);

if __name__ == '__main__':
    utf_list=list(t1_3755.union(latin62));
    # from_list(utf_list,fnt,"chs3755_en_uncase_digits");
    root_v2="/home/lasercat/ssddata/dictsv2/";
    # alljp=[os.path.basename(i) for i in glob.glob(os.path.join(root_v2,"*jp*"))];
    # allkr = [os.path.basename(i) for i in glob.glob(os.path.join(root_v2, "*kr*"))];
    # all=alljp+allkr
    fnt = "/home/lasercat/ssddata/metamk/NotoSansCJKsc-Regular.otf";

    all=["dabkrmlt"];
    # fnt = "/home/lasercat/ssddata/metamk/";

    for k in all:
        from_v2(root_v2,fnt,k)


