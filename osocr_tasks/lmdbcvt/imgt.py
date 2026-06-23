import glob
import os;
import shutil;

import cv2;
import numpy as np;

from neko_sdk import libnorm
from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper



class imgt_lmdbexport:
    def setup(this,**kwargs):
        pass;

    def handle_line(this,im,line):
        pass;

    def handle_file(this,im,gt):
        im=cv2.imread(im);
        with open(gt,"r") as fp:
            for l in fp:
                this.handle_line(im,l.strip());

    def add_ds_by_pairs(this,impaths,gt_paths):
        for i in range(len(impaths)):

            try:
                this.handle_file(impaths[i], gt_paths[i]);
                pass;
            except:
                print("error on ", impaths[i], gt_paths[i]);
    def add_ds(this,basepath,index):
        keys=[];
        with open(os.path.join(basepath,index), "r") as fp:
            for l in fp:
                keys.append(l.strip());
        impaths=[];
        gtpaths=[];
        for k in keys:
            imname = os.path.join(basepath, k + ".jpg");
            gtname = os.path.join(basepath, "gt_" + k + ".txt");
            gtpaths.append(gtname);
            impaths.append(imname);
        this.add_ds_by_pairs(impaths,gtpaths)

    def end_adding(this):
        this.db.end_this();
    def __init__(this,dst):
        shutil.rmtree(dst,True);
        os.makedirs(dst);
        this.db=im_lmdb_wrapper(dst);
        # this.spc=torch.load("spcmlt.pt");
        pass;