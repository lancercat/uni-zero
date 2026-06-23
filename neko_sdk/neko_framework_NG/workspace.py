from multiprocessing import Queue as mpQueue

import numpy as np
import torch
import copy
from neko_sdk.log import fatal, warn, tree_view, tree_view_prefixed,collapse_tree
from collections.abc import Iterable
from PIL import Image
# a workspace is where objection, intermediate data, memories, and logging_subs data go.
from neko_sdk.log import warn,info,fatal
import multiprocessing

class neko_workspace:
    def __init__(this,input_dict=None,epoch_idx=0,batch_idx=0,devices=None):
        if(input_dict is None):
            input_dict={};
        # something you can drop after forward pass, gets with read and write.
        # Consider using data queue if there is no need for grad passing, so you can actually run each part asynchornizely.
        # Still, we DONOT do sanity check, you are on your own.
        this.inter_dict=input_dict;
        # the default device, only used when the module does not know what it is on.

        if(devices is None):
            devices=["cuda"]; # use cuda by default.
        # All devices involved.
        this.devices = devices;
        # main device is always at position 0.
        this.device=devices[0];
        # Objectives, BP starts here.
        this.objdict={};
        # Logz
        this.logdict={}; # we don't want it to print

        # Your epoch index. Feel free to use this as a criteria to enable or disable functions.
        this.epoch_idx=epoch_idx;
        # Your batch index. Feel free to use this as a criteria to enable or disable functions.
        this.batch_idx=batch_idx;
    def get_iter(this):
        return this.batch_idx;
    def get_epoch(this):
        return this.epoch_idx;
    # no value copy, symbolink all contents to new dict---
    def shallow_fork_dict(this,sdict):
        ddict={};
        for k in sdict.keys():
            ddict[k]=sdict[k];
        return ddict;
    def shallow_fork(this):
        dst=this.empty_like(this);
        dst.inter_dict=this.shallow_fork_dict(this.inter_dict);
        dst.logdict=this.shallow_fork_dict(this.logdict);
        dst.objdict=this.shallow_fork_dict(this.objdict);
        return dst;
    @classmethod
    def empty_like(cls,workspace):
        return neko_workspace(epoch_idx=workspace.epoch_idx,batch_idx=workspace.batch_idx,devices=workspace.devices);
    def has(this,name):
        return name in this.inter_dict;
    def refdev(this,name):
        if(this.has(name)):
            return this.inter_dict[name].device;
        return this.devices[0];
    def get(this, name):
        if(not this.has(name)):
            tv=this.treeview_interdict();
            ptv=this.treeview_interdict(name);
            tvs = this.treeview_interdict_shallow();
            print(tvs)
            fatal("????",name,"not in interdict");
        return this.inter_dict[name];
    def maybe_get(this, name):
        if(not this.has(name)):
            return None;
        return this.inter_dict[name];
    def get_copy(this, name):
        if(not this.has(name)):
            fatal("????")
        return copy.deepcopy(this.inter_dict[name]) ;
    def alias(this,src,dst):
        this.add(dst,this.inter_dict[src]);
    def get_list(this, names):
        return [this.inter_dict[name] for name in names];

    def check_nan(this):
        for k in this.inter_dict:
            e=this.get(k);
            if(type(e) is torch.Tensor):
                if(torch.isnan(e).any()):
                    print(k);
    def add(this,name,value,allow_override=False):
        if(name is None):
            return ; # do no add something to nothing
        if(name in this.inter_dict and allow_override ==False):
            fatal(name+" already exists!!!!");
        # assert not this.has(name)
        this.inter_dict[name]=value;
    def append_add(this,name,value):
        if(not this.has(name)):
            this.inter_dict[name]=[];
        this.inter_dict[name].append(value);
    def stackiftensor(this,name,dim=0):
        if(torch.is_tensor(this.inter_dict[name][0])):
            this.inter_dict[name]=torch.stack(this.inter_dict[name],dim=dim).contiguous();
    def add_loss(this,name,value):
        assert name not in this.objdict
        this.objdict[name]=value;
    def get_log(this, name):
        return this.logdict[name];
    def add_log(this,name,value):
        assert name not in this.logdict
        this.logdict[name]=value;
    def add_log_image(this,name,value):
        assert name not in this.logdict["images"];
        this.logdict["images"][name]=value;
    def add_log_lines(this, name, value):
        assert name not in this.logdict["texts"];
        this.logdict["texts"][name] = value;
    # if we dequeue from multiple queues.

    def add_dict_as_is(this, input_dict):
        for k, v in input_dict.items():
            this.add(k, v);
    @classmethod
    # Recursive helper function to handle nested structures
    def contigous_recursive(cls,item):
        # Case 1: Item is a single tensor.
        if isinstance(item, torch.Tensor):
            return item.contiguous();
        # Case 2: Item is a dictionary.
        elif isinstance(item, dict):
            new_dict = {}
            for k, v in item.items():
                new_dict[k] = cls.contigous_recursive(v)
            return new_dict
        elif isinstance(item, (np.ndarray, Image.Image)):
            return item
        # Case 3: Item is an iterable (e.g., list, tuple).
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            # Handle list of lists, list of tensors, etc.
            return [cls.contigous_recursive(sub_item) for sub_item in item]
        # Case 4: Item is a non-tensor, non-iterable (e.g., str, int).
        else:
            return item
    @classmethod
    # Recursive helper function to handle nested structures
    def move_to_device_recursive(cls,item,device):
        # Case 1: Item is a single tensor.
        if isinstance(item, torch.Tensor):
            if device is not None:
                return item.to(device)
            return item
        # Case 2: Item is a dictionary.
        elif isinstance(item, dict):
            new_dict = {}
            for k, v in item.items():
                new_dict[k] = cls.move_to_device_recursive(v,device)
            return new_dict
        elif isinstance(item, (np.ndarray, Image.Image)):
            return item
        # Case 3: Item is an iterable (e.g., list, tuple).
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            # Handle list of lists, list of tensors, etc.
            return [cls.move_to_device_recursive(sub_item,device) for sub_item in item]
        # Case 4: Item is a non-tensor, non-iterable (e.g., str, int).
        else:
            return item

    def add_dict(this, input_dict, device=None):
        """
        Recursively adds key-value pairs to a 'this' object, moving tensors
        and nested data structures containing tensors to a specified device.
        """
        for k, v in input_dict.items():
            this.add(k, this.move_to_device_recursive(v,device));

    # debugging APIs, handy if you have break points

    @classmethod
    def filter_dict(cls,item_dict,ks_or):
        rdic={};
        for k in item_dict:
            for fk in ks_or:
                if(k.find(fk)!=-1):
                    rdic[k]=item_dict[k];
                    break;
        return rdic;

    def filter_interdict(this, ks_or):
        return this.filter_dict(this.inter_dict, ks_or);

    def treeview_interdict_shallow(this):
        tree = {};
        for k in this.inter_dict:
            nodes = k.split("-");
            stree = tree
            for n in nodes:
                if (n not in stree):
                    stree[n] = {};
                stree = stree[n];
            stree[k] = k;
        return collapse_tree(tree);

    def treeview_interdict(this,prfx=None):
        if(prfx is None):
            return tree_view(this.inter_dict)
        else:
            return tree_view_prefixed(this.inter_dict,prfx);

    # deprecating
    def make_subspace_interdict(this,interdict_mapping,ss):
        for k in interdict_mapping:
            t=this.get(k);
            if(interdict_mapping[k]["device"] is not None):
                t=t.to(interdict_mapping[k]["device"]);
            ss.add(interdict_mapping[k]["name"],t);
        return ss;

    def simple_fetch_interdict_as_subspace(this,interdict_keys=None,device=None,registeration_name=None):
        imap={};
        ss=neko_workspace();
        if(interdict_keys is not None):
            for k in interdict_keys:
                imap[k]={"name":k,"device":device};
        else:
            for k in this.inter_dict:
                imap[k]={"name":k,"device":device};

        ss=this.make_subspace_interdict(imap,ss);

        return ss;

    def DBG_FI(this,ks_or):
        return this.filter_dict(this.inter_dict,ks_or);


    # since the most bu

from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict

# They will be changeable by agents, but just cannot be dropped after each iteration.
class neko_environment:
    def replace_queue(this,name,q=None,maxsz=2):
        if(q is None):
            # USS Quincy
            ctx = multiprocessing.get_context("forkserver");
            q=ctx.Queue(maxsize=maxsz);
        this.queue_dict[name]=q;
    def enque(this,qname,what):
        # info("putting into", qname);
        this.queue_dict[qname].put(what);
        # info("put into", qname);

    def deque(this,qname):
        # info("getting from",qname);
        data=this.queue_dict[qname].get();
        # info("got from",qname);
        return data;
    # probably a died API.
    # You know I am just a cat and thus cannot remember everything I wrote...
    def view(this,mod_cvt_dict,queue_dict):
        vmodset=this.modset.view(mod_cvt_dict);

        pass;
    def save_mods(this):
        this.modset.save_necessary(this.epoch_idx,this.batch_idx);
    # Drop queues so it can be shared to another thread.
    def warp_ref(this):
        return neko_environment(assets_dict=this.assets_dict,modset=this.modset);
    def after_wrap(this,e):
        this.assets_dict=e.assets_dict;
        this.modset=e.modset;


    def __init__(this,assets_dict=None,queue_dict=None,modset:neko_module_opt_setNG=None):
        this.modset=modset;
        if(modset is not None):
            this.module_dict=get_modular_dict(modset);
        else:
            this.module_dict={};
        if(assets_dict is None):
            assets_dict={};
        if(queue_dict is None):
            queue_dict={};
            this.queue_dict = queue_dict;
        else:
            this.queue_dict = queue_dict;
            for k in queue_dict:
                if(queue_dict[k] is None):
                    warn("detected undefined queue: "+ k+ " defining");
                    this.replace_queue(k);
        this.batch_idx=0;
        this.epoch_idx=0;
        # something other than modules
        this.assets_dict=assets_dict;
        # blocking queues, for async uses.
        # please drop grad before doing anything to it.

    # since most business we have with it is to call one of the modules...
    # here is a shortcut.
    def __call__(this,name, *args, **kwargs):
        return this.module_dict[name](*args,**kwargs);
