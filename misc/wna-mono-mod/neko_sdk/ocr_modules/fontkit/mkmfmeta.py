import os.path
import shutil

import numpy as np

from osocrNG.lsctNG.libtrdg.mask_making_engine import neko_string_mask_generator
from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgr
from multiprocess.pool import Pool
from neko_sdk.neko_framework_NG.data.supportdb.support_lmdb_builder import neko_support_db_builder

import os.path

import cv2
import torch
import numpy as np


def addfnt(g,mgr,f,pool,root=None):
    for i in range(0,len(mgr.meta["fnt_charset"][f]),100):
        cs=mgr.meta["fnt_charset"][f][i:i+100];
        if(root is not None):
            fn=os.path.join(root,f);
        else:
            fn=f;
        ms, __= g.try_drive_para([fn for _ in range(len(cs))],cs,pool);
        sstr=os.path.basename(f);
        for m,c in zip(ms,cs):
            if(m is not None):
                i=cv2.resize(np.array(m)[:,:,0],(64,64));
                b.add_im_under_utf(i, c, None,None,sstr);

            else:
                print(c,"claimed to be supported by ",fn,"but not");
                pass;
                # print("empty",c,"for font",f);
        pass;
def is_noto_regular(f):
    fn=os.path.basename(f);
    return (fn.find("Noto")!=-1 and fn.find("Regular")!=-1 and fn.find("Sans")!=-1);

hdict={};


mgr = fntmgr();
mgr.tryload("/home/lasercat/ssddata/synth_lsct/cache/finfo251105.pt",
            "/home/lasercat/ssddata/synth_lsct/");
g=neko_string_mask_generator({
    neko_string_mask_generator.PARAM_size:64,
    neko_string_mask_generator.PARAM_orientations: [0]
});
pool=Pool(12);
shutil.rmtree("/run/media/lasercat/writebuffer/tmp/glyphdb_utf_260202/",ignore_errors=True);
os.makedirs("/run/media/lasercat/writebuffer/tmp/glyphdb_utf_260202/");
b=neko_support_db_builder("/run/media/lasercat/writebuffer/tmp/glyphdb_utf_260202/");

for f in mgr.meta["fnt_charset"]:
    if(f=="main"):
        continue;
    # if(is_noto_regular(f)):
    addfnt(g,mgr,f,pool,"/home/lasercat/ssddata/synth_lsct/");
pass;

# for f in mgr.meta["fnt_charset"]:
#     if(f=="main"):
#         continue;
#     if(is_noto_regular(f)):
#         addfnt(g,mgr,f,pool);
#     else:
#         pass;
# pass;
# for f in mgr.meta["fnt_charset"]:
#     if(f=="main"):
#         continue;
#     if(is_noto_regular(f)):
#         pass;
#     else:
#         addfnt(g,mgr,f,pool);
# pass;

b.end_this();
