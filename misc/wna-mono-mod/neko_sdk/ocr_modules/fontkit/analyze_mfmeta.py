import os.path

import cv2
import torch
import numpy as np
import tqdm

from neko_sdk.neko_framework_NG.data.supportdb.support_utf_im_lmdb import neko_support_utf_to_im
from neko_sdk.ocr_modules.renderlite.librender_mk3 import render_lite_mk3_nl


def find_sim_chars(froot,charset,maxutf=5):
    hashidx=torch.load(os.path.join(froot,"hash_index.pt"),weights_only=False);
    hashdet=torch.load(os.path.join(froot,"hash.pt"),weights_only=False);
    index_orig=torch.load(os.path.join(froot,"index.pt"),weights_only=False);
    index_filtered_path=os.path.join(froot,"filtered_index.pt");

    lhash=list(hashdet.keys());
    chardupcnt=[len(hashdet[x]) for x in hashdet];
    topcdupidx=np.argsort(chardupcnt)[::-1];
    topcduphash=[lhash[_] for _ in topcdupidx];
    topcdupcnt=[chardupcnt[_] for _ in topcdupidx];

    blklst=[];
    blkhash=[];
    index_filtered={};
    char_to_hash={};
    for i in tqdm.tqdm(range(len(topcdupidx))):
        hash=topcduphash[i];
        if(topcdupcnt[i]>maxutf):
            blklst+=hashidx[topcduphash[i]];
            blkhash.append(topcduphash[i]);
        else:
            for k in hashdet[hash]:
                if (k not in char_to_hash):
                    char_to_hash[k]=set();
                char_to_hash[k].add(hash);
    collateral=set();
    for c in charset:
        if(c in char_to_hash):
            for h in char_to_hash[c]:
                for k in hashdet[h]:
                    if(k not in collateral):
                        collateral.add(k);
                        if(k!=c):
                            print(k, "found for same render with",c);
    return collateral;

def filter_idx(froot,maxutf=5):
    hashidx=torch.load(os.path.join(froot,"hash_index.pt"),weights_only=False);
    hashdet=torch.load(os.path.join(froot,"hash.pt"),weights_only=False);
    index_orig=torch.load(os.path.join(froot,"index.pt"),weights_only=False);
    index_filtered_path=os.path.join(froot,"filtered_index.pt");

    lhash=list(hashdet.keys());
    chardupcnt=[len(hashdet[x]) for x in hashdet];
    topcdupidx=np.argsort(chardupcnt)[::-1];
    topcduphash=[lhash[_] for _ in topcdupidx];
    topcdupcnt=[chardupcnt[_] for _ in topcdupidx];


    blklst=[];
    blkhash=[];
    index_filtered={};
    for i in range(len(topcdupidx)):
        hash=topcduphash[i];
        if(topcdupcnt[i]>maxutf):
            blklst+=hashidx[hash];
            blkhash.append(topcduphash[i]);

    blklst=set(blklst);
    print("found", len(blklst), "renders being shared by more than",maxutf,"utf labels");
    for k in index_orig:
        nlst=[];
        for id, td,ed in index_orig[k]:
            if(id["image"] in blklst):
                continue;
            else:
                nlst.append((id,td,ed));
        if(len(nlst)):
            index_filtered[k]=nlst;
        else:
            print(k,"dropped for no legal rendering")
        pass;

def show_bad(froot):
    hashidx = torch.load(os.path.join(froot, "hash_index.pt"),weights_only=False);
    hashdet = torch.load(os.path.join(froot, "hash.pt"),weights_only=False);

    db = neko_support_utf_to_im({
        neko_support_utf_to_im.PARAM_root: froot
    });

    lhash = list(hashdet.keys());
    chardupcnt = [len(hashdet[x]) for x in hashdet];
    topcdupidx = np.argsort(chardupcnt)[::-1];

    topcduphash = [lhash[_] for _ in topcdupidx];
    topcdupk = [hashidx[_][0] for _ in topcduphash];

    topcdupcnt = [chardupcnt[_] for _ in topcdupidx];
    topcdupd = [hashdet[k] for k in topcduphash];
    for i in range(len(topcdupk)):
        im=db.fetch_img(topcdupk[i]);
        print(topcdupcnt[i],":",topcdupd[i] );
        cv2.imshow("meow",np.array(im));
        cv2.waitKey(0);


if __name__ == '__main__':
    wfroot = "/run/media/lasercat/writebuffer/tmp/glyphdb_utf_260202/";
    froot = "/run/media/lasercat/data/data_setstone/synth_lsct/glyphdb_utf_260202";
    from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
    from neko_sdk.ocr_modules.charset.etc_cset import latin62
    from neko_sdk.ocr_modules.charset.testing_csets import jpnhv
    from neko_sdk.ocr_modules.charset.kr_charset import korean
    find_sim_chars(wfroot,(korean.union(jpnhv)).difference(latin62).difference(t1_3755),5);
    # a=torch.load(os.path.join(froot,"filtered_index.pt"),weights_only=False);
    pass;


