import glob
import os;
import shutil;

import cv2;
import numpy as np;

from neko_sdk import libnorm
from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper

from osocr_tasks.lmdbcvt.imgt  import imgt_lmdbexport



class mlt_lmdbexport(imgt_lmdbexport):
    ign_lang={"Korean","Mixed"};
    ACC_lang=None;
    def setup(this,**kwargs):
        if "ACC_lang" in kwargs:
            this.ACC_lang=kwargs["ACC_lang"];
        if "ign_lang" in kwargs:
            this.ign_lang=kwargs["ign_lang"]
    def handle_line(this,im,line):
        fields=line.split(",",10);
        cords=np.array([int(i) for i in fields[:8]]).reshape([4,2]);

        lang=fields[8];
        content=fields[9];
        if(content.isdigit()):
            pass;

        if (content == "###"):
            return ;
        if (
                (this.ign_lang is not None)
                and
                (lang in this.ign_lang )
        ):
            print(lang);
            return;
        if(
            (this.ACC_lang is not None)
            and
            (lang not in this.ACC_lang)
        ):
            print(lang);
            return;
        im=libnorm.norm(im,cords, None);

        this.db.add_data_utf(im,content,lang);

class mlt_synth_export(imgt_lmdbexport):
    def setup(this,**kwargs):
        this.lang=kwargs["lang"];
    def handle_line(this,im,line):
        fields=line.split(" ",6);
        cords=np.array([int(i) for i in fields[:8]]).reshape([4,2]);
        lang=this.lang;
        content=fields[8][1:-1];
        if (content == "###"):
            return ;
        im=libnorm.norm(im,cords,libnorm.norm_sz(cords));

        this.db.add_data_utf(im,content,lang);
# export Welf mltfied lsvt dataset. For the raw dataset, go for lsvtcvt.py
class lsvt_lmdbexport(imgt_lmdbexport):
    def handle_line(this,im,line):
        cord,content=line.split(",\"",1);
        cords=[int(i) for i in cord.split(",")];
        try:
            cords=np.array(cords).reshape([4,2]);
        except:
            return ;
        lang="Unknown";
        content=content[:-1];
        if ("###" in content or "＃＃＃" in content):
            return ;
        im=libnorm.norm(im,cords,None);


        this.db.add_data_utf(im,content,lang);

class rctw_lmdbexport(imgt_lmdbexport):
    def handle_line(this,im,line):
        cord,content=line.split(",\"",1);
        fs=[int(i) for i in cord.split(",")]
        cords=fs[:-1];

        try:
            cords=np.array(cords).reshape([4,2]);
        except:
            return ;
        lang="Unknown";
        content=content[:-1];
        if (fs[-1]==1):
            print("skipped OOB",content);
            return ;
        im=libnorm.norm(im,cords,None);
        # cv2.imshow("baz",im);
        # cv2.waitKey(300);
        # if (im.shape[0] > im.shape[1]):
        #     return;
        # # if( im.shape[0]*0.1*len(content)>im.shape[1]):
        # cv2.imshow("baz",im);
        # print(content)
        # cv2.waitKey(300);
        # return ;

        this.db.add_data_utf(im,content,lang);

def make_mlt_train_jp(src, dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang="Japanese", ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();
def make_mlt_train_jp_II(src, dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang="Japanese", ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();
def make_mlt_valkr(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang="Korean", ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();
def make_mlt_valhindi(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang="Hindi", ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

def make_mlt_train_chlat(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Chinese","Latin"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();
def make_mlt_train_kr(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Korean"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

def make_mlt_train_bangla(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Bangla"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

def make_mlt_train_Hindi(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Hindi"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

def make_mlt_train_Korea(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Korean"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();


def make_mlt_train_Arabic(src,dst):
    expo = mlt_lmdbexport(dst);
    gt_files=[i for i in glob.glob(os.path.join(src,"*.txt"))];
    imgs=[i.replace("txt","jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.setup(ACC_lang=["Arabic"], ign_lang=None);
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

def make_rctw_train(src,dst):
    expo = rctw_lmdbexport(dst);
    gt_files = [i for i in glob.glob(os.path.join(src, "*.txt"))];
    imgs = [i.replace("txt", "jpg") for i in gt_files];
    # ign lang is used for situations like "I need anything but NOT bangala"
    expo.add_ds_by_pairs(imgs, gt_files)
    expo.end_adding();

if __name__ == '__main__':
    pass;

    # expo = mlt_lmdbexport("mltsyntchdb");
    # expo.add_ds("/media/lasercat/data/datasets/e2e/mltsynth/Chinese","all.index");
    # expo.end_adding();

    # expo=lsvt_lmdbexport("lsvtdb");
    # expo.add_ds("/home/lasercat/netdata/datasets/lsvt/img/","lsvt.index");
    # expo.end_adding();

    #
    # expo = mlt_lmdbexport("mltnkdb");
    # expo.add_ds("/home/lasercat/netdata/tmp/imgs","all.index");
    # expo.add_ds("/home/lasercat/netdata/e2e/mltsynth/Chinese", "all.index");
    # expo.end_adding();
    #

    #
    # expo=mlt_lmdbexport("mltchvaldb");
    # expo.setup(ACC_lang=["Chinese","Latin"],ign_lang=None);
    # expo.add_ds("/media/lasercat/data/datasets/tmp-val/imgs","all.index");
    # expo.end_adding();
    # #
    # expo=mlt_lmdbexport("mltchlatdbval");
    # expo.setup(ACC_lang=["Chinese","Latin"],ign_lang=None);
    # expo.add_ds("/media/lasercat/data/datasets/tmp-val/imgs","all.index");
    # expo.end_adding();
    # expo=rctw_lmdbexport("rctwdb");
    # expo.add_ds("/media/lasercat/DATA/rctw/rctw_train/rctwall","rctw.index");
    # expo.end_adding();
