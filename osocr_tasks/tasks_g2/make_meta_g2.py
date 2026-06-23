import glob
import os

import torch

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt;
from osocr_tasks.tasks_g2.dscsg2 import upgrade_pt


def index_fnts(fnts):
    odict={};
    pdict={};
    for fnt in fnts:
        try:
            cset=fntmgmt.get_charset(fnt);
            odict[os.path.basename(fnt)]=cset;
            pdict[os.path.basename(fnt)]=fnt;
            print(fnt,"OK");
        except:
            print("perhaps bad font", fnt);
    return pdict,odict;

#
# pdict,odict=index_fnts(fnts);
# torch.save([pdict,odict],"/homealter/lasercat/ssddata_extra/pamiremake/gfindex.pt");
#
#
# [pdict,odict]=torch.load("/homealter/lasercat/ssddata_extra/pamiremake/gfindex.pt");
# cs=set(torch.load("/home/lasercat/ssddata/dicts/dab3791WT.pt")["chars"]);
# for n in odict:
#     if(len(odict[n].intersection(cs))==len(cs)):
#         print(pdict[n]);
#     elif(len(odict[n].intersection(cs))>100):
#         print(pdict[n],len(cs-odict[n]));
def upgrade_meta(src,dst):
    fnts=glob.glob("/home/lasercat/Downloads/g2/*.tt*")
    meta=torch.load(src);
    dmeta=upgrade_pt(meta,fnts);
    torch.save(dmeta,dst);
# upgrade_meta("/home/lasercat/ssddata/dicts/dab3791WT.pt","/home/lasercat/ssddata/dicts/dab2g3791WT.pt");
#upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl20_train/dict.pt","/home/lasercat/ssddata/ctwch/ctwfsl20_train/dict2g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dict.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dict2g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej2g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej50.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej502g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej100.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej1002g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej150.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej1502g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej200.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej2002g.pt");
upgrade_meta("/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej250.pt","/home/lasercat/ssddata/ctwch/ctwfsl_5_1eval/dictrej2502g.pt");
