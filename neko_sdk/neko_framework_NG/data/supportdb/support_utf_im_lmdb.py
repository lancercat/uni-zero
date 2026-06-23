# well lsctng nomore uses character based render, the meta will still use that, so we gonna but a unified interface for glyph storage....
# and plus this makes an entity database for object detection/ synthmaking as well, for whatever task.
import os.path
import random

import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.data.abstract_lmdb_loader import neko_abstract_lmdb_loader,neko_abstract_data_source_NG
import  numpy as np

class neko_support_utf_to_im(neko_abstract_lmdb_loader):
    CONST_INDX_FILE="index.pt";
    KEY_UTF="utf";
    KEY_id="id";
    def init_etc(this,para):
        this.utfd=torch.load(os.path.join(this.root,this.CONST_INDX_FILE),weights_only=False);
    def fetch_core(this, descp):
        ik=this.utfd[descp[this.KEY_UTF]][descp[this.KEY_id]];
        if(type(ik)==str):
            return this.fetch_img(ik);
        return this.fetch_img(ik[0]["image"]);
    def all_valid_indexes(this):
        alldscp=[];
        for u in this.utfd:
            alldscp+=[{this.KEY_UTF:u,this.KEY_id:_}  for _ in range(len(this.utfd[u]))];
        return alldscp;
class neko_support_utf_to_im_filtered(neko_abstract_lmdb_loader):
    CONST_INDX_FILE="filtered_index.pt";


class neko_support_utf_to_im_multi(neko_abstract_data_source_NG):
    PARAM_root_dict="root_dict";
    PARAM_root_names="root_names"; # defines the order
    KEY_UTF="utf";
    KEY_id="id";
    KEY_dsid="dsid";
    def get_core_fetcher(this,root):
        return neko_support_utf_to_im({neko_support_utf_to_im.PARAM_root: root});

    def setup(this,para):
        rd=neko_get_arg(this.PARAM_root_dict,para);
        this.roots=neko_get_arg(this.PARAM_root_names,para,list(rd.keys()));
        this.idd={};
        this.dss=[];
        for k in this.roots:
            this.idd[k]=len(this.dss);
            this.dss.append(this.get_core_fetcher(rd[k]));
        this.init_etc(para);
        pass;

    def init_etc(this,para):
        this.utfidx={};
        for i in range(len(this.dss)):
            for k in this.dss[i].utfd:
                if(k not in this.utfidx):
                    this.utfidx[k]=[];
                # k is here bcs it will need it to navigate the children
                this.utfidx[k]+=[{this.KEY_UTF:k,this.KEY_dsid:i,this.KEY_id:_} for _ in range(len(this.dss[i].utfd[k]))];

    def fetch_item(this, descp):
        u=descp[this.KEY_UTF];
        dsid=descp[this.KEY_dsid];
        id=descp[this.KEY_id];
        return np.array(this.dss[dsid].fetch_item({neko_support_utf_to_im.KEY_UTF:u,
                                         neko_support_utf_to_im.KEY_id:id}));
    def all_valid_indexes(this,):
        alldscp=[];
        for u in this.utfidx:
            alldscp+=[{this.KEY_UTF:u,this.KEY_dsid:_[this.KEY_dsid],this.KEY_id:_[this.KEY_id]}  for _ in range(len(this.utfidx[u]))];
        return alldscp;
class neko_support_utf_to_im_multi_filtered(neko_support_utf_to_im_multi):
    def get_core_fetcher(this,root):
        return neko_support_utf_to_im_filtered({neko_support_utf_to_im_filtered.PARAM_root: root});

class neko_zero_support_fetcher:
    PARAM_root_dict="root_dict";
    PARAM_root_names="root_names";
    def get_fetcher(this,rd):
        return neko_support_utf_to_im_multi({
            neko_support_utf_to_im_multi.PARAM_root_names: this.roots,
            neko_support_utf_to_im_multi.PARAM_root_dict: rd
        });

    # defines the order
    def __init__(this,params):
        rd = neko_get_arg(this.PARAM_root_dict, params);
        this.roots = neko_get_arg(this.PARAM_root_names, params, list(rd.keys()));
        this.data=this.get_fetcher(rd);
        this.utfidx=this.data.utfidx;
    def fetch(this,u,k):
        return [this.data.fetch_item(this.utfidx[u][0])];



class neko_random_support_fetcher(neko_zero_support_fetcher):
    PARAM_seed="seed";
    def __init__(this,params):
        super().__init__(params);
        this.seed=neko_get_arg(this.PARAM_seed,params,9);
        this.rng=random.Random(this.seed);
    def fetch(this,u,k):
        if(u not in this.utfidx):
            return None; # if there is no such thing, don't make proto for it

        return [this.data.fetch_item(_) for _ in random.choices(this.utfidx[u], k=k)]
class neko_random_support_fetcher_filtered(neko_zero_support_fetcher):
    def get_fetcher(this,rd):
        return neko_support_utf_to_im_multi_filtered({
            neko_support_utf_to_im_multi_filtered.PARAM_root_names: this.roots,
            neko_support_utf_to_im_multi_filtered.PARAM_root_dict: rd
        });
class neko_first_k_support_fetcher(neko_zero_support_fetcher):
    def __init__(this,params):
        super().__init__(params);
        pass;
    def fetch(this,u,k):
        return [this.data.fetch_item(_) for _ in this.utfidx[u][:k]];

