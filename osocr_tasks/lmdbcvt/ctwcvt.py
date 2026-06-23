
import json
import os;
import shutil

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
def handle_word_gt(wdict):
    text=""
    kps=[];
    for ch in wdict:
        text+=ch["text"];
        kps+=ch["polygon"];
    if len(wdict)>1:
        magicord=np.array(wdict[-1]["polygon"])-np.array(wdict[0]["polygon"])
        div=magicord[0][0];
        if div==0:
            div=0.0009;
        angguess=np.arctan(magicord[0][1]/div)/np.pi*180;
        contors=fix_rrect(cv2.minAreaRect(np.array(kps).astype(np.int32)),angguess);
    else:
        return text,None;
    return text,contors;


def handle_file_gt(fdict):
    texts,boxes=[],[];
    for item in fdict["annotations"]:
        text,box=handle_word_gt(item);
        texts.append(text);
        boxes.append(box);
    return texts,boxes;

def subimage(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape,flags=cv2.INTER_CUBIC  )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

def handle_file(fdict,dataroot,db):
    file_name = os.path.join(dataroot,fdict["file_name"]);
    im = cv2.imread(file_name);
    texts,boxes=handle_file_gt(fdict);
    # cv2.namedWindow("meow",0);
    for b in range(len(boxes)):
        if(boxes[b] is None):
            continue;
        center,(w,h),theta=boxes[b];
        clip=subimage(im,center,theta,int(w),int(h));
        # print(texts[b]);
        if(clip.shape[0]<5 or clip.shape[1]<=5):
            continue;
        # cv2.imshow("meow",clip);
        # cv2.waitKey(0);
        db.add_data_utf(clip,texts[b],"Chinese");

def make_ctw(trpath,dataroot,dst):
    jdicts = []

    shutil.rmtree(dst,True);

    db=im_lmdb_wrapper(dst);

    with open(trpath,"r") as fp:
        for l in fp:
            jdict=json.loads(l);
            jdicts.append(jdict);
    for fid in range(len(jdicts)):
        if(fid%0x71==0):
            print(fid, " of ", len(jdicts));
        handle_file(jdicts[fid],dataroot,db);
    db.end_this();

if __name__ == '__main__':
    ROOT="/home/lasercat/ssddata/deploy";

    PATHS = {
        "artroot": ROOT + "/art",
        "ctwtrgtroot": ROOT + "/ctw/gtar/train.jsonl",
        "ctwvagtroot": ROOT + "/ctw/gtar/val.jsonl",
        "ctwtrimroot": ROOT + "/ctw/jpgs",
        "mltroot": ROOT + "/mlt/real",
        "mltsynthchroot": ROOT + "/mlt/Chinese",
        "rctwtrroot": ROOT + "/rctw_train/train",
        "lsvttrjson": ROOT + "/lsvt/train_full_labels.json",
        "lsvttrimgs": ROOT + "/lsvt/imgs/",
        "fntpath": [ROOT + "/NotoSansCJK-Regular.ttc"],
        "dictroot": ROOT + "/deployedlmdbs/dicts",
        "desroot": ROOT + "/deployedlmdbs"
    }
    # rawctwtr=os.path.join(PATHS["desroot"],"ctwdbtr");
    # make_ctw(PATHS["ctwtrgtroot"],
    #          PATHS["ctwtrimroot"],
    #          rawctwtr);
    rawctwva=os.path.join(PATHS["desroot"],"ctwdbval");
    make_ctw(PATHS["ctwvagtroot"],
             PATHS["ctwtrimroot"],
             rawctwva);