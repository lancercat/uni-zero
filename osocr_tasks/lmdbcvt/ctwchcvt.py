import json;
import os;
import shutil;

import cv2;
import numpy as np;

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper


def final_fix(cou,angguess):
    c,(w,h),t=cou;
    if angguess>60:
        return c,(h,w),t-90;
    if angguess < -60:
        return c, (h, w), t + 90;
    return cou;

def fix_rrect(cou,angguess):
    c,(w,h),t=cou;
    if(np.abs(angguess-t)>120):
        return None;
    if(np.abs(angguess-t)>60):
        if(angguess>t):
            t+=90;
            return final_fix((c,(h,w),t),angguess);
        else:
            t-=90;
            return final_fix((c,(h,w),t),angguess);
    return final_fix(cou,angguess);
def handle_char_gt(cdict):
    rect = cdict["adjusted_bbox"];
    text = cdict["text"];
    attr=cdict["attributes"]
    return text,rect,attr;

def handle_word_gt(wdict):
    texts=[];
    boxes=[];
    attrs=[];
    for ch in wdict:
        t,b,a=handle_char_gt(ch);
        texts.append(t);
        boxes.append(b);
        attrs.append(a);
    return texts,boxes,attrs;


def handle_file_gt(fdict):
    texts,boxes,attrs=[],[],[];
    for item in fdict["annotations"]:
        text,box,attr=handle_word_gt(item);
        texts+=text;
        boxes+=box;
        attrs+=attr
    return texts,boxes,attrs;

def subimage(image, box):
    l,t,w,h=box;
    l=int(l);
    r=int(l+w);
    t=int(t);
    b = int(t+h);
    return image[t:b,l:r,:];


def handle_file(fdict,dataroot,db):
    file_name = os.path.join(dataroot,fdict["file_name"]);
    im = cv2.imread(file_name);
    texts,boxes,attrs=handle_file_gt(fdict);
    # cv2.namedWindow("meow",0);
    for b in range(len(boxes)):
        if(boxes[b] is None):
            continue;
        box=boxes[b];
        clip=subimage(im,box);
        # print(texts[b]);
        if(clip.shape[0]<5 or clip.shape[1]<=5):
            continue;
        if((b+1)%39==0):
            cv2.imshow("meow",clip);
            cv2.waitKey(10);
        db.adddata_kv({"image":clip},{"label":texts[b],"lang":"Chinese","attr":str(attrs[b])},{});

def make_ctwch(trpath,dataroot,dst):
    jdicts = []
    with open(trpath, "r") as fp:
        for l in fp:
            jdict = json.loads(l);
            jdicts.append(jdict);
    shutil.rmtree(dst, True);
    db = im_lmdb_wrapper(dst);
    for fid in range(len(jdicts)):
        if (fid % 0x71 == 0):
            print(fid, " of ", len(jdicts));
        handle_file(jdicts[fid],dataroot,db);
    db.end_this();

