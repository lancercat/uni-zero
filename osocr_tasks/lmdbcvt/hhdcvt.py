import glob
import os.path
import shutil

import cv2
import tqdm

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
def make_ethopian_lmdb(root,dst):
    splits=["train_raw","test_rand","test_18th"];
    for s in splits:
        p=os.path.join(dst,"_"+s);
        shutil.rmtree(p,ignore_errors=True);
        os.makedirs(p,exist_ok=True);
        ir=im_lmdb_wrapper(p);
        af=glob.glob(os.path.join(root,s,"image","*.png"));
        d={};
        with open(os.path.join(root,s,"image_text_pairs_"+s+".csv"),"r") as fp:
            for p in fp:
                k,v=p.strip().split(",");
                d[k]=v
        print(len(af));
        for f in af:
            ir.add_data_utf(cv2.imread(f), d[os.path.basename(f)], "Ethopian");

if __name__ == '__main__':
    make_ethopian_lmdb("/run/media/lasercat/writebuffer/mooster-ex2/HHD-Ethiopic","/run/media/lasercat/320-eccv/results/ethopian/")


