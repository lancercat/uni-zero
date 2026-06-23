# well lsctng nomore uses character based render, the meta will still use that, so we gonna but a unified interface for glyph storage....
# and plus this makes an entity database for object detection/ synthmaking as well, for whatever task.
import os.path
import random

import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.lmdb_wrappers.lmdb_wrapper import lmdb_wrapper
from neko_sdk.neko_framework_NG.data.abstract_lmdb_loader import neko_abstract_lmdb_loader,neko_abstract_data_source_NG
import  numpy as np
import hashlib
from neko_sdk.log import fatal,info,warn


def filter_idx(froot,maxutf=5):
    hashidx=torch.load(os.path.join(froot,"hash_index.pt"));
    hashdet=torch.load(os.path.join(froot,"hash.pt"));
    index_orig=torch.load(os.path.join(froot,"index.pt"),weights_only=False);
    index_filtered_path=os.path.join(froot,"filtered_index.pt");

    lhash=list(hashdet.keys());
    chardupcnt=[len(hashdet[x]) for x in hashdet];
    topcdupidx=np.argsort(chardupcnt)[::-1];
    topcduphash=[lhash[_] for _ in topcdupidx];
    topcdupcnt=[chardupcnt[_] for _ in topcdupidx];


    blklst=[];
    index_filtered={};
    for i in range(len(topcdupidx)):
        if(topcdupcnt[i]>maxutf):
            blklst+=hashidx[topcduphash[i]];
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
    torch.save(index_filtered,index_filtered_path);


def hash_numpy_array_md5(array: np.ndarray) -> str:
    """
    Hashes a 2D numpy array using MD5.

    Args:
        array: The 2D numpy array to hash.

    Returns:
        The MD5 hash as a hexadecimal string.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Convert the array to a byte string.
    # The tobytes() method creates a bytes object from the array's data.
    # We also include the dtype and shape in the bytes to ensure
    # arrays with the same data but different types/shapes produce different hashes.
    array_bytes = array.tobytes()
    metadata_bytes = f"{array.dtype}-{array.shape}".encode('utf-8')
    full_bytes = array_bytes + metadata_bytes

    # Create an MD5 hash object
    hasher = hashlib.md5()

    # Update the hasher with the byte string
    hasher.update(full_bytes)

    # Get the hexadecimal representation of the hash
    return hasher.hexdigest()


class neko_support_db_builder(lmdb_wrapper):
    CONST_INDX_FILE="index.pt";
    CONST_HASH_FILE="hash.pt";
    CONST_HASH_IDX_FILE="hash_index.pt";

    def __init__(this,lmdb_dir,start=0,resume=False):
        super().__init__(lmdb_dir,start);
        this.lmdb_dir=lmdb_dir;
        this.idx_file=os.path.join(lmdb_dir,this.CONST_INDX_FILE);
        this.hash_file = os.path.join(lmdb_dir, this.CONST_HASH_FILE);
        this.hash_idx_file = os.path.join(lmdb_dir, this.CONST_HASH_IDX_FILE);
        this.utfd={};
        this.hdict={};
        this.hidx={};
        if(resume):
            this.utfd=torch.load(this.idx_file);
            this.hdict=torch.load(this.hash_file);
            this.hidx=torch.load(this.hash_idx_file);

    def add_stub(this,image,hash,utf,mask, attr,srcstr):
        if(utf not in this.utfd):
            this.utfd[utf]=[];
        if(hash not in this.hidx):
            this.hidx[hash]=[];
        td={"label":utf};
        if(srcstr is not None):
            td["source"]=srcstr;
        if(len(attr)):
            td["attr"]=attr; # lang is a type of attr now.
        id={"image":image}
        if(mask is not None):
            id["polymsk"]=mask;
        iks,tks,rks=this.adddata_kv(id,td,{});
        this.utfd[utf].append((iks,rks,tks));
        this.hidx[hash].append(iks["image"]);

    def add_im_under_utf(this, image,utf,mask=None,attr=None,srcstr=None):
        if attr is None:
            attr=[];
        if(len(image.shape)==3):
            h = hash_numpy_array_md5(image[:,:,0]);
        else:
            h = hash_numpy_array_md5(image);

        if (h not in this.hdict):
            this.hdict[h] = {};
            this.hdict[h][utf] = [srcstr];
            this.add_stub(image,h, utf,mask,attr, srcstr);
        else:
            if (utf in this.hdict[h]):
                this.hdict[h][utf].append(srcstr);
                #warn("EXACT SAME",utf, "already in", this.hdict[h][utf][:3], "skipping, not duplicating from", srcstr);
            else:
                warn(utf,"from:",srcstr, "VISUALLY identical to these:", str(list(this.hdict[h].keys())[:6]), "adding nevertheless");
                this.hdict[h][utf] = [srcstr];
                this.add_stub(image,h, utf,mask,attr, srcstr);
            pass;
        pass;
    def end_this(this):
        torch.save(this.utfd,this.idx_file);
        torch.save(this.hdict,this.hash_file);
        torch.save(this.hidx,this.hash_idx_file);
        filter_idx(this.lmdb_dir);
        super().end_this();
