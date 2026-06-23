import regex
import torch;

from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt;
from neko_sdk.ocr_modules.renderlite.addfffh import refactor_meta, add_masters, finalize
from neko_sdk.ocr_modules.renderlite.lib_render import render_lite


def get_ds(root,filter=True,support_multiw=True):
    charset = {};
    db=neko_ocr_lmdb_mgmt(root,not filter,1000);
    for i in range(len(db)):
        _,t=db.getitem_encoded_im(i);
        try:
            for c in regex.findall(r'\X', t, regex.U):
                if(not support_multiw and len(c)>1) :
                    print("unsupported compond character", c);
                    continue;
                if(c not in charset):
                    charset[c]=0
                charset[c]+=1;
        except:
            print(t);
            pass;
        if(i%300==0):
            print(i, "of" , len(db),"ds",root)
    return charset;

#
# servants="QWERTYUIOPASDFGHJKLZXCVBNM";
# masters="qwertyuiopasdfghjklzxcvbnm";
# another example to make it fancier(multiple centers)
# servants=["qf1","wf2",'qf2','wf1'];
# masters="qwqw";
def makept_core( chrset,fonts,font_ids,protodst, servants="QWERTYUIOPASDFGHJKLZXCVBNM", masters="qwertyuiopasdfghjklzxcvbnm", space=None,support_multiw=True):

    engine = render_lite(os=84,fos=32);
    meta=engine.render_core(chrset,['[s]'],fonts,font_ids,False);
    meta=refactor_meta(meta,unk=len(chrset)+len(['[s]']));
    # inject a shapeless UNK.
    meta["protos"].append(None)
    meta["achars"].append("[UNK]")
    if(len(masters)):
        add_masters(meta,servants,masters);
    # add_masters(meta,servants,masters);
    meta=finalize(meta);
    torch.save(meta,protodst);
    return chrset
def makept(dataset, font, protodst, xdst, blacklist, servants="QWERTYUIOPASDFGHJKLZXCVBNM", masters="qwertyuiopasdfghjklzxcvbnm", space=None,support_multiw=True):
    if(dataset is not None):
        if(space is not None):
            chrset=list(set(xdst.union(get_ds(dataset,False,support_multiw=support_multiw))).difference(blacklist).intersection(space));
        else:
            chrset=list(set(xdst.union(get_ds(dataset,False,support_multiw=support_multiw))).difference(blacklist));
    else:
        chrset=list(set(xdst).difference(blacklist));
    font_ids=[0 for c in chrset];
    return makept_core(chrset,font,font_ids,protodst,servants,masters,space,support_multiw);

from glob import glob
import os
def scanfolder_and_add_pt(root,font,xdst,blacklist,depth=2):
    d=os.path.join(root,"*");
    dslist=[];
    for i in range(depth):
        dslist_d = glob(os.path.join(d,"*.mdb"));
        d=os.path.join(d,"*");
        dslist+=[os.path.dirname(i) for i in dslist_d]
    for data in dslist:
        makept(data,font,os.path.join(data,"dict.pt"),xdst,blacklist,servants="",masters="");
