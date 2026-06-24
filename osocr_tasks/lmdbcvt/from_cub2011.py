import copy
import glob
import os.path
import shutil

import cv2
import tqdm

from osocr_tasks.tasks_gen3.char_classifier.mkdict import v3_dict_from_list
import json;
from neko_sdk.neko_framework_NG.data.supportdb.support_lmdb_builder import neko_support_db_builder

KEY_MSK="mask";
KEY_IMG="image";
KEY_DOC_NAME="doc";
KEY_INST_ID="inst_id";
IDFR="_";

def mk_utf(cn,cntrn):
    return cn+IDFR+cntrn;


def add_file_tree(root,split_dict):
    chars=glob.glob(os.path.join(root,"*"));
    for c in chars:
        cn=os.path.basename(c);
        if (cn not in split_dict):
            split_dict[cn]={};
        cntrs=glob.glob(os.path.join(c,"*"));
        for cntr in cntrs:
            cntrn=os.path.basename(cntr);
            if(cntrn not in split_dict[cn]):
                split_dict[cn][cntrn]={};
            instances=glob.glob(os.path.join(cntr,"img*png"));
            for i in instances:
                iname = os.path.basename(i);
                _,_,docid,seqid=iname.split(".")[0].split(IDFR);
                giid=int(docid)*100000+int(seqid);
                idic={
                    KEY_IMG: i,
                    KEY_DOC_NAME: docid,
                    KEY_INST_ID: seqid
                };
                split_dict[cn][cntrn][giid]=idic;

    return split_dict;


def sorted_dict(src_dict):
    for cn in src_dict:
        for cntrn in src_dict[cn]:
            orded={k:src_dict[cn][cntrn][k]for k in sorted(src_dict[cn][cntrn])};
            src_dict[cn][cntrn]=orded;
    pass;
def add_to_support(srcdict, supdict,removal=False):
    dstdict=copy.deepcopy(srcdict);
    for cn in srcdict:
        if (cn not in supdict):
            supdict[cn] = {};
        for cntrn in srcdict[cn]:
            if cntrn not in supdict[cn]:
                supdict[cn][cntrn] = {};
            if(len(supdict[cn][cntrn])):
                continue;
            sks = sorted(list(srcdict[cn][cntrn].keys()));
            supdict[cn][cntrn][sks[0]] = srcdict[cn][cntrn][sks[0]];
            if removal:
                dstdict[cn][cntrn]={k:dstdict[cn][cntrn][k] for k in sks[1:]};
            if (len(dstdict[cn][cntrn]) == 0):
                del dstdict[cn][cntrn];
        if (len(dstdict[cn]) == 0):
            del dstdict[cn];

    return dstdict,supdict;


# add one first sample of each task
def mkteval_support_all_trcplus_tevalfirst(train_dict,teval_dict):
    sdict={};
    _,sdict=add_to_support(train_dict,sdict,False);
    return add_to_support(teval_dict,sdict,True);

def to_utf_list_and_masterdict(srcdict):
    mdict={};
    ulist=[];
    for cn in srcdict:
        forms=sorted(list(srcdict[cn].keys()));
        utfs=[mk_utf(cn,f) for f in forms];
        for u in utfs:
            mdict[u]=utfs[0];
        ulist+=utfs;
    return mdict,ulist


# we don't wish to duplicate db if we don't have to.
def export_sdict(srcdict,name,meta_root="/home/lasercat/ssddata/dicts_v3/"):
    mdict,ulist=to_utf_list_and_masterdict(srcdict)
    v3_dict_from_list(ulist,name,meta_root,master_dict=mdict);

def export_lmdb(srcdict,support_root="/home/lasercat/ssddata/"):
    b = neko_support_db_builder(support_root);
    for cn in srcdict:
        for cntrn in srcdict[cn]:
            sks = sorted(list(srcdict[cn][cntrn].keys()));
            for k in sks:
                sdic=srcdict[cn][cntrn][k]
                b.add_im_under_utf(cv2.imread(sdic[KEY_IMG]),mk_utf(cn,cntrn),None,[],None);
    b.end_this();

def allim(srcdict):
    an=[];
    for cn in srcdict:
        for cntrn in srcdict[cn]:
            sks = sorted(list(srcdict[cn][cntrn].keys()));
            for k in sks:
                sdic=srcdict[cn][cntrn][k]
                an.append(sdic[KEY_IMG]);
    return an;

def cache_cub_split(dataset,root,split):
    dsname="cub2011_";
    tempp=root+split;
    shutil.rmtree(tempp,ignore_errors=True);
    os.makedirs(tempp)
    d=dataset[split];
    lcnt={};
    for eid in tqdm.tqdm(range(d.num_rows)):
        example=dataset[split][eid];
        image = example["image"];
        bbox = example["bbox"];
        label=example["label"];
        cropped_image = image.crop(bbox);
        if(label not in lcnt):
            os.makedirs(os.path.join(tempp, str(label), "0"));
            lcnt[label]=0;
        save_path = os.path.join(tempp,str(label),"0", "img_doc_0_"+str(lcnt[label])+".png");
        lcnt[label]+=1;
        cropped_image.save(save_path);
        pass;
def export_cub_split(ataset,root,split):
    dsname="cub2011_";
    dd={};
    tempp=root+split;
    dd=add_file_tree(tempp,dd);
    sorted_dict(dd);
    export_sdict(dd,dsname+split);
    export_lmdb(dd,dsname+split);
    _,sd=mkteval_support_all_trcplus_tevalfirst(dd,dd);
    export_lmdb(sd, dsname + split+"_support");

if __name__ == '__main__':
    import datasets
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    root="/run/media/lasercat/320-eccv/results/cub/"
    dataset = datasets.load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
    split="train";
    export_cub_split(dataset, root, split);

    split="test";
    export_cub_split(dataset, root, split);

