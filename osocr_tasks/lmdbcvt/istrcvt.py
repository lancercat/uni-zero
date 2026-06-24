import glob
import os.path
import shutil

import cv2
import tqdm

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
def make_indic_lmdb(root,dst):
    langs=["odia"]#[os.path.basename(f) for f in glob.glob(os.path.join(root,"*"))];
    splits=["train","val","test"];
    for l in langs:
        for s in splits:
            p=os.path.join(dst,l+"_"+s);
            shutil.rmtree(p,ignore_errors=True);
            os.makedirs(p,exist_ok=True);
            ir=im_lmdb_wrapper(p);
            with open(os.path.join(root,l,s,s+"_images.txt"),"r") as fp:
                fs=[os.path.join(root,l,_.strip()) for _ in fp];
            with open(os.path.join(root, l, s, s + "_labels.txt"), "r") as fp:
                gts = [_.strip() for _ in fp];
            for f,g in tqdm.tqdm(zip(fs,gts)):
                ir.add_data_utf(cv2.imread(f),g,l);

if __name__ == '__main__':
    make_indic_lmdb("/run/media/lasercat/writebuffer/mooster-ex2/istr-10th-Sep/unarxiv","/run/media/lasercat/320-eccv/results/indiclmdb/")


