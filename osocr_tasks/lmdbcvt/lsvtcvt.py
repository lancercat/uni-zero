
import json
import os;
import shutil

import cv2;
import numpy as np;

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper


def handle_word_gt(wdict):
    text=wdict["transcription"]
    ps=np.array(wdict["points"]);
    xmax=ps[:,0].max();
    xmin = ps[:,0].min();
    ymax=ps[:,1].max();
    ymin = ps[:,1].min();


    return text,[xmin,xmax,ymin,ymax];


def handle_file_gt(fdict):
    texts,boxes=[],[];
    for item in fdict:
        if(item['illegibility']==True):
            continue;
        text,box=handle_word_gt(item);
        texts.append(text);
        boxes.append(box);
    return texts,boxes;

def handle_file(fname,fdict,dataroot,db):
    pass;
    file_name = os.path.join(dataroot,fname);
    im = cv2.imread(file_name);
    texts,boxes=handle_file_gt(fdict);
    # # cv2.namedWindow("meow",0);
    for b in range(len(boxes)):
        if(boxes[b] is None):
            continue;
        xmin,xmax,ymin,ymax=boxes[b];
        clip=im[ymin:ymax,xmin:xmax];
        # print(texts[b]);
        if(clip.shape[0]<5 or clip.shape[1]<=5):
            continue;
        # cv2.imshow("meow",clip);
        # cv2.waitKey(0);
        db.add_data_utf(clip,texts[b],"Chinese");

def makelsvt(trpath,dataroot,dst):

    shutil.rmtree(dst,True);

    db=im_lmdb_wrapper(dst);
    fid=0
    with open(trpath,"r") as fp:
        jdict=json.load(fp);
        for fn in jdict:
            if(fid%0x71==0):
                print(fid, " of ", len(jdict));
            handle_file(fn+".jpg",jdict[fn],dataroot,db);
            fid+=1;
    db.end_this();
if __name__ == '__main__':
    makelsvt(
             "/media/lasercat/backup/deploy/lsvt/train_full_labels.json",
             "/media/lasercat/backup/deploy/lsvt/imgs",
             "/media/lasercat/backup/deployedlmdbs/lsvtdb")
