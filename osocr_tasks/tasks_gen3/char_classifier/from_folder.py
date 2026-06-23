import glob
import json
import os

import cv2
import numpy as np

from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
import torch
from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
from neko_sdk.neko_framework_NG.libmeta.file_names import FN

def split_trainval(rd,val_ratio):
    ans=list(rd.keys());
    ids=np.argsort([os.path.basename(k) for k in ans]);
    ks=[ans[i] for i in ids];
    p=int(val_ratio*len(ks));
    vks=ks[:p];
    tks=ks[p:];
    trd={};
    vald={};
    for k in vks:
        vald[k]=rd[k];
    for k in tks:
        trd[k]=rd[k];
    return trd,vald;

def mkdict_labeled_folder(root,pfx,trans_dict=None):
    files=glob.glob(os.path.join(root,"*","*."+pfx));
    labels=[os.path.basename(os.path.dirname(f)) for f in files];
    rd={};
    for f,l in zip(files,labels):
        if(trans_dict is not None):
            l=trans_dict[l];
        rd[f]=l;
    return rd;

def arm_lmdb(rd,lmdbroot,lang):
    os.makedirs(lmdbroot,exist_ok=True);
    db=im_lmdb_wrapper(lmdbroot);
    idxd={};
    for f in rd:
        idxs=db.add_data_utf(cv2.imread(f),rd[f],lang)
        for lab in tokenize(rd[f]):
            if(lab not in idxd):
                idxd[lab]=[];
            idxd[lab].append({"imf":f,
                           "dbk":idxs});
    torch.save(idxd,os.path.join(lmdbroot,"idx.pt"));
    db.end_this();
def mk_dft_meta(rd):
    ls=set();
    for f in rd:
        for t in tokenize(rd[f]):
            ls.add(t);
    tdict={};
    i=0;
    utfs=[];
    for l in ls:
        tdict[l] = i;
        tdict[i] = l;
        i += 1;
        utfs.append(l);
    return utfs,tdict


def from_labeled_folder(root,lmdbroot,trans_dict,lang="unknown"):
    rd=mkdict_labeled_folder(root,"png",trans_dict);
    trd,vald=split_trainval(rd,0.2);
    with open("train.json","w") as fp:
        json.dump(trd,fp);
    with open("val.json", "w") as fp:
        json.dump(vald, fp);

    arm_lmdb(trd,os.path.join(lmdbroot,"train"), lang);
    arm_lmdb(trd,os.path.join(lmdbroot,"val"), lang);
    utfs,tdict=mk_dft_meta(rd);
    torch.save({
        FN.KEY_UTF_LST:utfs,
        FN.KEY_IDS: list(range(len(utfs))),
     },os.path.join(lmdbroot,FN.ID_META));
    torch.save(tdict, os.path.join(lmdbroot, FN.TDICT));



if __name__ == '__main__':
    with open("/run/media/lasercat/writebuffer/histcomp/training/coco.json","r") as fp:
        coco=json.load(fp);
    trd={};
    for i in range(len(coco ["categories"])):
        trd[str(i)]=coco ["categories"][i]["name"];
    from_labeled_folder("/run/media/lasercat/writebuffer/histcomp/patches/",
                        "/run/media/lasercat/writebuffer/histcomp/lmdb/",
                        trd,
                        "old_latin");

