import regex
import torch;

from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt;
from neko_sdk.ocr_modules.renderlite.addfffh import refactor_meta, add_masters, finalize
from neko_sdk.ocr_modules.renderlite.lib_render import render_lite


#
def get_ds(root,filter=True):
    charset = {};
    db=neko_ocr_lmdb_mgmt(root,not filter,1000);
    for i in range(len(db)):
        _,t=db.getitem_encoded_im(i);
        try:
            for c in regex.findall(r'\X', t, regex.U):
                if(c not in charset):
                    charset[c]=0
                charset[c]+=1;
        except:
            print(t);
            pass;
        if(i%300==0):
            print(i, "of" , len(db),"ds",root)
    return charset;
def get_ds_from_group(roots,filter=True):
    charset = {};
    for root in roots:
        csd=get_ds(root, filter);
        for c in csd:
            if (c not in charset):
                charset[c] = 0
            charset[c] += csd[c];
    return charset;

#
# servants="QWERTYUIOPASDFGHJKLZXCVBNM";
# masters="qwertyuiopasdfghjklzxcvbnm";
# another example to make it fancier(multiple centers)
# servants=["qf1","wf2",'qf2','wf1'];
# masters="qwqw";
def makeptg2core(chrset,font,protodst,    servants="QWERTYUIOPASDFGHJKLZXCVBNM",masters="qwertyuiopasdfghjklzxcvbnm",whitelist=None):

    engine = render_lite(os=84,fos=32);
    font_ids=[[0] for c in chrset];
    meta=engine.render_coreg2_mem(chrset,['[s]'],font,font_ids,False);
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

def makeptg2(dataset,font,protodst,xdst,blacklist,    servants="QWERTYUIOPASDFGHJKLZXCVBNM",masters="qwertyuiopasdfghjklzxcvbnm",whitelist=None):
    if(dataset is not None):
        if(whitelist is not None):
            chrset=list(set(xdst.union(get_ds(dataset,False))).difference(blacklist).intersection(whitelist));
        else:
            chrset=list(set(xdst.union(get_ds(dataset,False))).difference(blacklist));
    else:
        chrset=list(set(xdst).difference(blacklist));
    return makeptg2core(chrset,font, protodst, servants, masters, whitelist);
def makeptg2_fromgroup(datasets,font,protodst,xdst,blacklist,    servants="QWERTYUIOPASDFGHJKLZXCVBNM",masters="qwertyuiopasdfghjklzxcvbnm",whitelist=None):
    allc=get_ds_from_group(datasets,False);
    if(whitelist is not None):
        chrset=list(set(xdst.union(allc)).difference(blacklist).intersection(whitelist));
    else:
        chrset=list(set(xdst.union(allc)).difference(blacklist));

    return makeptg2core(chrset,font, protodst, servants, masters, whitelist);

def upgrade_pt(meta,fonts):
    engine = render_lite(os=84,fos=32);
    meta_= engine.render_coreg2_mem(meta["chars"],['[s]'],fonts,
                                    [list(range(len(fonts))) for c in meta["chars"]],False);
    meta["protos"] = meta_["protos"];
    return meta;


from glob import glob
import os
def scanfolder_and_add_pt(root,font,xdst,blacklist):
    dslist=glob(os.path.join(root,"*"));
    for data in dslist:
        makeptg2(data,font,os.path.join(data,"dict.pt"),xdst,blacklist);

