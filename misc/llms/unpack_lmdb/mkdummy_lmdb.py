from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
import glob
import os
import shutil
import cv2
import os.path

import cv2
import numpy as np

from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
def dump(src,dst):
    os.makedirs(dst,exist_ok=True);
    h=neko_im_label_lmdb_holder({neko_im_label_lmdb_holder.PARAM_root:src,neko_im_label_lmdb_holder.PARAM_vert_to_hori:-1});
    for vi in h.all_valid_indexes():
        rd=h.fetch_core(vi);
        cv2.imwrite(os.path.join(dst,str(vi["id"])+".png"),cv2.cvtColor(np.array(rd["image"]),cv2.COLOR_RGB2BGR));
def dump_txt(src,dst):
    os.makedirs(dst,exist_ok=True);
    h=neko_im_label_lmdb_holder({neko_im_label_lmdb_holder.PARAM_root:src,neko_im_label_lmdb_holder.PARAM_vert_to_hori:-1});
    for vi in h.all_valid_indexes():
        rd=h.fetch_core(vi);
        with open(os.path.join(dst,str(vi["id"])+".txt"),"w+") as fp:
            fp.writelines(rd["label"]);


def quick_lmdb_dummygt(image_path,dst,lang="None",imgpfix="png"):
    samples_im=glob.glob(os.path.join(image_path,"*."+imgpfix));
    shutil.rmtree(dst,True);
    db=im_lmdb_wrapper(dst);
    for iname in samples_im:
        img=cv2.imread(iname);
        if(img is None):
            print("bad", iname);
            continue;
        anno="meow";
        db.add_data_utf(img,anno,lang);
    db.end_this();
    dump(dst,os.path.join(dst,"imgs"));


if __name__ == '__main__':
    # quick_lmdb_dummygt("/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/ssddata/athenaNG/yi/","/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/ssddata/dummyi/")
    # dump("/run/media/lasercat/writebuffer/tmp/synthyi_5k/","/run/media/lasercat/writebuffer/tmp/synthyi_5k/imgs")
    dump_txt("/run/media/lasercat/writebuffer/tmp/synthyi_5k/","/home/lasercat/project313_collection/synthyi5k/gt")