import os.path
import shutil

import cv2

from neko_sdk.neko_framework_NG.data.supportdb.support_utf_im_lmdb import neko_random_support_fetcher,neko_first_k_support_fetcher

def dumpdb( split,    sroot="/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/ssddata/ecr-v3R/", droot="/home/lasercat/writebuffer/ecr_support/",):

    dst=os.path.join(droot,split);
    f=neko_first_k_support_fetcher({
        neko_first_k_support_fetcher.PARAM_root_names: ["ecr-v3R"],
        neko_first_k_support_fetcher.PARAM_root_dict:{
            "ecr-v3R": os.path.join(sroot,"ecr_v3_"+split+"_support")
        }
    });
    shutil.rmtree(dst,ignore_errors=True);
    os.makedirs(dst);
    for u in f.utfidx:
        r=f.fetch(u,1);
        cv2.imwrite(os.path.join(dst,u+".jpg"),r[0]);
        pass;

if __name__ == '__main__':

    dumpdb("tra");
    dumpdb("test");
    dumpdb("val");