import copy
import glob
import os.path

import cv2
import torch

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
    totcntr=0;
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
                    KEY_MSK: os.path.join(cntr,iname.replace("img","mask")),
                    KEY_DOC_NAME: docid,
                    KEY_INST_ID: seqid
                };
                totcntr+=1
                split_dict[cn][cntrn][giid]=idic;

    return split_dict;
def filter_by_docid(dst_dict,src_dict,whitelist):
    for cn in src_dict:
        if(cn not in dst_dict):
            dst_dict[cn]={};
        for cntrn in src_dict[cn]:
            if cntrn not in dst_dict[cn]:
                dst_dict[cn][cntrn]={};
            for idick in src_dict[cn][cntrn]:
                idic=src_dict[cn][cntrn][idick];
                if idic[KEY_DOC_NAME] in whitelist:
                    dst_dict[cn][cntrn][idick]=idic;
            if (len(dst_dict[cn][cntrn]) == 0):
                del dst_dict[cn][cntrn];
        if (len(dst_dict[cn]) == 0):
            del dst_dict[cn];


def sorted_dict(src_dict):
    for cn in src_dict:
        for cntrn in src_dict[cn]:
            orded={k:src_dict[cn][cntrn][k]for k in sorted(src_dict[cn][cntrn])};
            src_dict[cn][cntrn]=orded;
    pass;
def add_to_support(srcdict, supdict,removal=False,known=None):
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
            if known is not None:
                if (cn in known):
                    print(cn, cntrn, "taxed for new shape");
            if (len(dstdict[cn][cntrn]) == 0):
                del dstdict[cn][cntrn];
        if (len(dstdict[cn]) == 0):
            del dstdict[cn];
    print("-----")

    return dstdict,supdict;


# add one first sample of each task
def mkteval_support_all_trcplus_tevalfirst(train_dict,teval_dict):
    sdict={};
    _,sdict=add_to_support(train_dict,sdict,False);
    return add_to_support(teval_dict,sdict,True,known=train_dict);

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
def export_sdict(srcdict,name,meta_root="/run/media/lasercat/writebuffer/tmp/dicts_v3/"):
    mdict,ulist=to_utf_list_and_masterdict(srcdict)
    v3_dict_from_list(ulist,name,meta_root,master_dict=mdict);

def export_lmdb(srcdict,support_root="/home/lasercat/ssddata/"):
    b = neko_support_db_builder(support_root);
    stat={}
    for cn in srcdict:
        for cntrn in srcdict[cn]:
            sks = sorted(list(srcdict[cn][cntrn].keys()));
            for k in sks:
                sdic=srcdict[cn][cntrn][k];
                img=cv2.imread(sdic[KEY_IMG]);
                utf=mk_utf(cn,cntrn);
                if(utf not in stat):
                    stat[utf]=[];
                stat[utf].append(sdic[KEY_IMG]);
                b.add_im_under_utf(img,utf,cv2.imread(sdic[KEY_MSK]),[],None);
    b.end_this();
    return stat;

def allim(srcdict):
    an=[];
    for cn in srcdict:
        for cntrn in srcdict[cn]:
            sks = sorted(list(srcdict[cn][cntrn].keys()));
            for k in sks:
                sdic=srcdict[cn][cntrn][k]
                an.append(sdic[KEY_IMG]);
    return an;
def count(splitd,excd):
    t=0;
    for i in splitd:
        if(i in excd):
            continue;
        t+=len(splitd[i]);
    return t;
def reduce_dic(d,ks):
    rd={};
    for k in ks:
        rs={};
        for c in d[k]:
            rk=c.split("_")[0];
            if(rk not in rs):
                rs[rk]=[];
            else:
                pass;
            rs[rk]+=d[k][c];
        rd[k]=rs;
    return rd;
if __name__ == '__main__':
    a=torch.load("/run/media/lasercat/writebuffer/tmp/ecr-v3-R/sta.pt");
    ar=reduce_dic(a,a);
    ttd={};
    for k in ar:
        if(k!="train"):
            ttd[k + "-novel"] = count(ar[k], ar["train"]);
        ttd[k]=count(ar[k],{});
    pass;

    #
    trd={};
    ted = {};
    vad = {};
    vad=add_file_tree("/run/media/lasercat/writebuffer/tmp/ecr-v3-R/Validation",vad);
    trd=add_file_tree("/run/media/lasercat/writebuffer/tmp/ecr-v3-R/Training",trd);
    ted= add_file_tree("/run/media/lasercat/writebuffer/tmp/ecr-v3-R/Testing", ted);


    sorted_dict(trd);
    sorted_dict(ted);
    sorted_dict(vad);

    ted,tes=mkteval_support_all_trcplus_tevalfirst(trd,ted);

    vad,vas=mkteval_support_all_trcplus_tevalfirst(trd,vad);

    _,trs=mkteval_support_all_trcplus_tevalfirst(trd,trd);


    teims=allim(ted);

    trims=allim(trd);

    valims=allim(vad);


    export_sdict(trd,"ecr_v3_train");
    export_sdict(tes, "ecr_v3_test");
    export_sdict(vas, "ecr_v3_val");
    astat={};
    astat["train"]=export_lmdb(trd,"ecr_v3_train");

    astat["test_samp"]=export_lmdb(ted,"ecr_v3_test");
    astat["test_supp"]=stat_tes=export_lmdb(tes, "ecr_v3_test_support");

    astat["val_samp"]=export_lmdb(vad, "ecr_v3_val");
    astat["val_supp"]=export_lmdb(vas, "ecr_v3_val_support");

    # this does not make sense--- but if you want to test on training set.
    export_lmdb(trs, "ecr_v3_tra_support");

    torch.save(astat,"/run/media/lasercat/writebuffer/tmp/ecr-v3-R/sta.pt")
    #
    with open("support_samples_testing.json", 'w') as f:
        json.dump(tes, f, indent=4, sort_keys=True)
    with open("support_samples_validation.json", 'w') as f:
        json.dump(vas, f, indent=4, sort_keys=True)
    with open("support_samples_train.json", 'w') as f:
        json.dump(trs, f, indent=4, sort_keys=True)
    pass;
