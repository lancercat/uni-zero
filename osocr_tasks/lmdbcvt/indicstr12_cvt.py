import glob
import json;
import os;
import shutil;

import cv2;
import numpy as np;
import tqdm

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
from osocr_tasks.ds_paths import get_indicstr12_synth,INDIC_STR12
SPLIT_train="train";
SPLIT_test="test";
SPLIT_val="val";

def get_synth_idx(dataroot,lang,split):
    if(split==SPLIT_train):
        return glob.glob(os.path.join(dataroot,lang,"*clean*_2M_train_gt.txt"))[0];
    elif(split==SPLIT_test):
        return glob.glob(os.path.join(dataroot,lang,"*_halfM_test_gt.txt"))[0];
    else:
        return glob.glob(os.path.join(dataroot,lang,"*valid_gt.txt"))[0];

def make_indicstr12_synth(index,dataroot,dst,lang):
    with open(index, "r") as fp:
        fs= [l.strip() for l in fp];
    shutil.rmtree(dst, True);
    db = im_lmdb_wrapper(dst);
    for l in tqdm.tqdm(fs):
        fn,gt=l.split("\t");
        i=cv2.imread(os.path.join(dataroot,lang,fn));
        if(i is None):
            print("bad", fn);
        else:
            db.add_data_utf(i,gt,lang);
    db.end_this();

if __name__ == '__main__':
    DROOT="/run/media/lasercat/writebuffer/mooster-ex2/indicstr-12/indicstr12";
    # "Kannada", "Malayalam", "Marathi",
    IN_QUESTION="Meitei_Manipuri";
    INDIC_STR12TOGO = ["Odia", "Punjabi", "Tamil", "Telugu", "Urdu"];

    for lang in INDIC_STR12TOGO:
        for split in [SPLIT_train,SPLIT_test,SPLIT_val]:
            ifile=get_synth_idx(DROOT,lang,split);
            dfile=get_indicstr12_synth(lang,split);
            make_indicstr12_synth(ifile,DROOT,dfile,lang);
