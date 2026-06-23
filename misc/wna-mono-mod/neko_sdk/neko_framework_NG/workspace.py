from multiprocessing import Queue as mpQueue

import torch

from neko_sdk.log import fatal,warn

# a workspace is where objection, intermediate data, memories, and logging_subs data go.
class neko_workspace:
    def __init__(this,input_dict=None,local_asset_dict=None,subspaces_dict=None,epoch_idx=0,batch_idx=0,device="cuda",devices=None):
        if(input_dict is None):
            input_dict={};
        # we are deprecating this stuff. It will however be here for a few moments before we decide we figure if keeping dynamic modules functional feasible (and if we can just abuse interdict if exception occures).
        if(subspaces_dict is None):
            subspaces_dict={};
        if(local_asset_dict is None):
            local_asset_dict={};


        # something you can drop after forward pass, gets with read and write.
        # Consider using data queue if there is no need for grad passing, so you can actually run each part asynchornizely.
        # Still, we DONOT do sanity check, you are on your own.
        this.inter_dict=input_dict;
        # some modules/assets dynamically generated for the exact iteration..,
        this.local_asset_dict=local_asset_dict;
        # subspaces, if modules and variables are needed.
        this.subspaces_dict=subspaces_dict;
        # the default device, only used when the module does not know what it is on.
        this.device=device;
        # Objectives, BP starts here.
        this.objdict={};
        # Logz
        this.logdict={
            "images":{

            },
            "texts":{

            },
        }; # we don't want it to print

        # All devices involved.
        if(devices is None):
            this.devices=[device];
        else:
            this.devices=devices;
            # always put you main device at 0th.
            assert this.device==devices[0];
        # Your epoch index. Feel free to use this as a criteria to enable or disable functions.
        this.epoch_idx=epoch_idx;
        # Your batch index. Feel free to use this as a criteria to enable or disable functions.
        this.batch_idx=batch_idx;
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
        if(registeration_name is not None):
            this.subspaces_dict[registeration_name]=ss;
        return ss;


    def get(this, name):
        return this.inter_dict[name];
    def alias(this,src,dst):
        this.add(dst,this.inter_dict[src]);
    def get_list(this, names):
        return [this.inter_dict[name] for name in names];
    def get_asset(this,name):
        return this.local_asset_dict[name];

    def add_asset(this, name, value):
        assert name not in this.local_asset_dict
        this.local_asset_dict[name] = value;

    def add(this,name,value):
        assert name not in this.inter_dict
        this.inter_dict[name]=value;
    def append_add(this,name,value):
        if(name not in this.inter_dict):
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

    def add_log(this,name,value):
        assert name not in this.logdict
        this.logdict[name]=value;

    # since the most bu

from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict

# They will be changeable by agents, but just cannot be dropped after each iteration.
class neko_environment:
    def replace_queue(this,name,q=None):
        if(q is None):
            # USS Quincy
            q=mpQueue(maxsize=8);
        this.queue_dict[name]=q;
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
