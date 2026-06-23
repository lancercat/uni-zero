import copy
import os

import torch
from torch import nn
from torch.nn import parallel as trnp
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.log import fatal,info,warn

from typing import Optional

# Also, NG modular is no more callable. It is now a mere container & controller
# use a bogo to warp it up and bind it's mapping it to a device, which you may find one solution in
# Data parallel is still too much if we want fancy controls, let's hope we will support that in 3G.
# We are not going to implement p2p in NG. maybe 3G will support centerless FL.

# you can also set a server if you want to do FL
# the server is a dict, include address, server-side name and sync frequency.

# before we link into hanazo or automerge, let's do it locally...

class neko_local_modular_repo:
    PARAM_mod_name="mod_name";
    PARAM_forks="forks";

    FORK_total_itr="total_iter";
    FORK_last_commit="last_head";

    COMMIT_branch_name="branch_name";
    COMMIT_eff_iter="effective_iteration"; # effective_iteration, tells how much the weight see
    COMMIT_raw_iter="raw_iteration"; # raw_iteration, tells how much iters the client contributed
    COMMIT_weight_dict="weights";

    def hash(this,weight_dict,name,iter):
        return name+str(iter);
        pass;

    def effective_iter(this,exclude="NEP_skipped_NEP"):
        eff=0;
        for h in this.forks:
            if(h==exclude):
                continue;            # don't count twice.
            eff+=this.forks[h][this.FORK_total_itr];
        return eff;
    # be it basic fedavg, genetic stuff,  flwr over lan, or some fancy stuff like automerge.
    # here is a basic fedavg


    # fed avg like--- Nep knows what exact dogoo is this....
    # it seriously? favors the last commiter.. not a good sign but let's put it here.
    # I mean if it works we don't fix it.
    # if we have the vram I guess I will just make it fedavg..
    # or kiss, just monika (i mean, EMA)
    def merge(this,branch_weight,branch_name,branch_raw_iter):
        # oeff=this.effective_iter(branch_name);
        # teff=oeff+branch_raw_iter;
        # wthis=branch_raw_iter/teff;
        # wbase=1-wthis;
        wthis=0.01;
        wbase=0.99;
        # you say the commiting branch's weights already have a portion of the commit albeit in a older version?
        # hu cares? we are not the Nep darn methmatical ppl. we approximate it when we can.
        # in all seriousness, i just don't want to have the extra overhead to actually keep a head of
        # each branch and actually re-average or compute the incrementation (-w_{old}*oldparam+w_new*newparam).
        # Berzelius (fat, bcs you don't want to do that on CPU) is not tap water.
        for k in this.head:
            this.head[k]=this.head[k]*wbase+branch_weight[k]*wthis;
        return

    def commited_to_stub(this,weight_dict, name,branch_raw_iter,branch_eff_iter):
        # if you have nothing in your head, take whatever even its nonsense
        if(this.head is None):
            this.head=weight_dict;
        # if you have something in your head, don't take nonsense.
        elif(branch_eff_iter==0):
            return ;
        else:
            this.merge(weight_dict,branch_eff_iter);
        this.head_hash = this.hash(this.head, name, this.effective_iter());
        if name not in this.forks:
            this.forks[name] = {
                this.FORK_total_itr: branch_raw_iter,
            }
        return ;


    def commited_to(this, commit):
        this.commited_to_stub(commit[this.COMMIT_weight_dict],
                              commit[this.COMMIT_branch_name],commit[this.COMMIT_raw_iter],commit[this.COMMIT_eff_iter]);



    # so the model updates it's own effective iter upon receiving the weights---
    # it includes the training iterations from it's peers so the number will be larger
    def pulled_from(this):
        return this.head_hash,this.effective_iter();


    # the historical statdicts will not be saved...
    # Too much disk consumption...
    def __init__(this,param):
        this.head=None;
        this.head_hash=None;
        this.commits={};
        this.forks={};


    def sync(this,commit):
        this.commited_to(commit);
        return this.pulled_from();
    # save the repo to disk, duh.
    def save(this,path):
        pass;

# save_each is controlled by something else, on a trainer level.

class neko_modular_NG:
    PARAM_mod_type="mod_type";
    PARAM_mod_param="mod_param";
    PARAM_server="server";
    PARAM_save_each="save_each"; # maybe it IS a bad idea to save hyper frequently
    PARAM_save_path="save_path";
    PARAM_name="name";
    PARAM_opt="opt";
    PARAM_opt_engine="opt_engine";
    PARAM_opt_lr = "learning_rate";
    PARAM_opt_weight_decay="weight_decay";
    PARAM_itrkey="itrkey";
    PARAM_pretrain="pretrain";
    PARAM_sched_override="optim_sched_override";
    PARAM_tags="tags";
    PARAM_max_grad_norm="max_grad_norm";
    SOP={
        "adam":torch.optim.Adam,
        "adadelta" :torch.optim.Adadelta,
        "rmsprop":torch.optim.RMSprop
    }

    # let me think about how do we setup multi severs (disks, collections) to sync to.
    # however, having more than one collection can mess up the
    def __init__(this,save_path,name,
                 module:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 optimizer_sched:torch.optim.lr_scheduler.LRScheduler,
                 server:Optional[neko_local_modular_repo]=None,
                 pretrain=None,
                 save_each=20000,  # don't save that frequently to you hard drive.
                 # don't merge pretrain here--- it will caz a weird situation where initialization order matters.
                 # just use some dead branches--- and NOT use the pretrain option together with
                 tags=None, # this thing in the future will allow you to freeze a certain group of modules across agents.
                                # now it does not do a nepping thing.
                 max_grad_norm=20
                 ):
        super(neko_modular_NG, this).__init__()
        if (server is not None) and (pretrain is not None):
            fatal("if you are using an FL server, make sure you load pretrain / continue your training there...")
        this.save_path=save_path;
        this.model=module;
        this.optimizer=optimizer;
        this.optimizer_sched=optimizer_sched;
        this.name=name;
        this.save_each=save_each;
        this.effective_iter=0;
        this.raw_iter=0;
        this.tags={};
        this.max_grad_norm=max_grad_norm;

        if(tags is not None):
            this.tags=copy.copy(tags);
        # yup, we leave save_each=0 a special case where the model is trainable but not saved
        if(save_each<0):
            this.freeze();
            info(this.name, "is frozen");
        this.server=server;
        if(this.save_path is not None and pretrain is not None):
            info(this.name, "loads from", pretrain);
            this.model.load_state_dict(torch.load(pretrain));
        # well, if your server does automerge magic, just do it!
        if(this.server is not None):
            # if the server want to give you a merged pretrain, just nep darn accept it.
            sd,this.effective_iter=this.server.sync(name,this.model.state_dict(),{
                this.server.FORK_total_itr:this.effective_iter
            });
            this.model.load_state_dict(sd);

    # we will not support synchornized multi-gpu in the future...
    def get_mod(this):
        return this.model;


    def detach(this):
        this.model.requires_grad_(False)
    def attach(this):
        this.model.requires_grad_(True)

    def train(this,training=True):
        this.model.train(training);
    def eval(this):
        this.model.eval();
    def normgrad(this):
        if this.save_each>0:
            nn.utils.clip_grad_norm_(this.model.parameters(), this.max_grad_norm, 2)


    def to(this,dev="cuda"):
        this.model.to(dev);
    def place(this,mapping_dict):
        this.model.to(mapping_dict["NEP_main_NEP"]);
    def bfloat16(this):
        this.model.bfloat16();
    def next_epoch(this):
        this.optimizer_sched.step();


    def freeze(this):
        this.model.requires_grad_(False);
    def unfreeze(this):
        this.model.requires_grad_(True);

        # config it but not buiding it yet (as some one may meddle with it before)



    @classmethod
    def get_default_optim(cls, params, lr=1.0, weight_decay=0.0005, sched_override=None, opte=None):
        optimizer = opte(params, lr=lr, weight_decay=weight_decay);
        if (sched_override is None):
            optimizer_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5], 0.3)
        else:
            optimizer_sched = sched_override["engine"](optimizer, **sched_override["params"]);
        return optimizer, optimizer_sched;

    @classmethod
    def get_default_NG_modular(cls, params):
        mod = params[cls.PARAM_mod_type](params[cls.PARAM_mod_param]);
        opt_cfg = neko_get_arg(cls.PARAM_opt, params, "NEP_skipped_NEP");
        save_each = neko_get_arg(cls.PARAM_save_each, params, 20000);
        save_path = neko_get_arg(cls.PARAM_save_path, params);
        itrk = neko_get_arg(cls.PARAM_itrkey, params, "TopNep");
        max_grad_norm=neko_get_arg(cls.PARAM_max_grad_norm,params,20);
        weight_decay = neko_get_arg(cls.PARAM_opt_weight_decay, params, 0.0005);
        learning_rate = neko_get_arg(cls.PARAM_opt_lr, params, 1.0);
        sched_override = neko_get_arg(cls.PARAM_sched_override, params, "NEP_skipped_NEP")
        tags=neko_get_arg(cls.PARAM_tags,params,"NEP_skipped_NEP");

        if (save_each <= 0):
            opt, opt_sched = None, None;
        elif (opt_cfg is None):
            if (len(list(mod.parameters())) == 0):
                opt, opt_sched = None, None
            else:
                opte = neko_get_arg(cls.PARAM_opt_engine, params, "NEP_skipped_NEP");
                opte = cls.SOP[opte];
                opt, opt_sched = cls.get_default_optim(mod.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                                       sched_override=sched_override, opte=opte);
        else:
            fatal("error");
        pretrain = neko_get_arg(cls.PARAM_pretrain, params, "NEP_skipped_NEP");
        modular = neko_modular_NG(
            save_path=save_path,
            name=params[cls.PARAM_name],
            module=mod,
            optimizer=opt,
            optimizer_sched=opt_sched,
            save_each=save_each,
            pretrain=pretrain,
            tags=tags,
        max_grad_norm=max_grad_norm
        );
        modular.load(None, itrk, pretrain is None);
        return modular;
    @classmethod
    def add_config_to_dict(cls, modcfgdict, name, mod_type, mod_param, optim_param,tags=None):
        if optim_param is None:
            optim_param= {
                cls.PARAM_save_each: 0,
                cls.PARAM_save_path: "NEP_skipped_NEP",
            };
        modcfgdict[name] =dict({
            cls.PARAM_name: name,
            cls.PARAM_mod_type:mod_type,
            cls.PARAM_mod_param:mod_param,
            **optim_param
        });
        if(tags is not None):
            modcfgdict[name][cls.PARAM_tags]=copy.copy(tags);
        return modcfgdict;

    # Save and load will in the far future be replaced with the fetch/merge/pull/push function.
    def load_from(this, root,name, itrkey, is_fresh=True):
        if (name == None):
            name = this.name;
        if (root == None):
            info("Module", name, "dose not support saving");
            return;

        p = os.path.join(root, name + itrkey + ".pth");
        op = os.path.join(root, name + itrkey + "_opt.pth");
        osp = os.path.join(root, name + itrkey + "_opt_sched.pth");
        try:
            this.model.load_state_dict(torch.load(p))
        except:
            try:
                this.model.load_state_dict(torch.load(p).state_dict());
                info("Module", this.name, "loaded as a hack");
            except:
                if (is_fresh):
                    warn("Module", this.name, "cannot load MODULE", "itr", p, ", starting fresh");
                else:
                    warn("Module", this.name, "cannot load", "itr", p, ", starting from upstream pretrain");

        try:
            this.optimizer.load_state_dict(torch.load(op));
        except:
            try:
                this.optimizer.load_state_dict(torch.load(op).state_dict());
            except:
                warn("Module", this.name, "cannot load OPTIMIZERS", "itr", op, ", starting fresh");
        try:
            this.optimizer_sched.load_state_dict(torch.load(osp));
        except:
            try:
                this.optimizer_sched.load_state_dict(torch.load(osp).state_dict());
            except:
                warn("Module", this.name, "cannot load OPTIMIZER_SCHED", "itr", osp, ", starting fresh");
        info("let me remind you that queue contents can not be restored for now");

    def load(this, name, itrkey, is_fresh=True):
        return this.load_from(this.save_path,name,itrkey,is_fresh);
    def save_as(this,root,key):
        torch.save(this.model.state_dict(), os.path.join(root,this.name+ key+".pth") );
        if(this.optimizer is not None):
            torch.save(this.optimizer.state_dict(), os.path.join(root,this.name+ key+"_opt.pth"));
        if(this.optimizer_sched is not None):
            torch.save(this.optimizer_sched.state_dict(), os.path.join(root,this.name+ key+"_opt_sched.pth"));
    def save_as_if_savable(this,root,nEpoch,batch_idx):
        if(this.save_each>0 ):
            if(batch_idx>0 or nEpoch==0):
                info("Saving", os.path.join(root, this.name)+'_E{}_I{}.pth'.format(nEpoch, batch_idx))
                this.save_as(root,'_E{}_I{}'.format(nEpoch, batch_idx));
            else:
                info("Saving", os.path.join(root, this.name)+ '_E{}.pth'.format(nEpoch));
                this.save_as(root, '_E{}'.format(nEpoch));

    def save_as_if_needed(this,root,nEpoch,batch_idx):
        if(this.save_each>0 and batch_idx%this.save_each==0):
            return this.save_as_if_savable(root,nEpoch,batch_idx);

    def force_save_savable(this,nEpoch,batch_idx):
        return this.save_as_if_savable(this.save_path,nEpoch,batch_idx);

    def save(this,nEpoch):
        if(this.save_each>0 ):
            this.save_as(this.save_path,'_E{}'.format(nEpoch));
            this.save_as(this.save_path,'latest');

    def save_if_needed(this,nEpoch,batch_idx):
        this.save_as_if_needed(this.save_path,nEpoch,batch_idx);
    def is_nan(this):
        is_nan = torch.stack([torch.isnan(p).any() for p in this.model.parameters()]).any()
        return is_nan;

    def commit(this, cur_itr):
        """
        Creates a snapshot of the current state on CPU and calculates iteration delta.
        Returns a dict containing the 'commit message'.
        """
        new_loc_itr = cur_itr - this.raw_iter
        this.raw_iter = cur_itr

        # Move state_dict to CPU to save GPU VRAM and keep the snapshot static
        # We use {k: v.cpu().clone() ...} to ensure a true detached copy
        cpu_sd = {k: v.cpu().clone() for k, v in this.model.state_dict().items()}

        commit_msg = {
            "state_dict": cpu_sd,
            "metadata": {
                "name": this.name,
                "iter_delta": new_loc_itr,
                "total_raw_iter": this.raw_iter,
                "tags": copy.deepcopy(this.tags)
            }
        }

        info(f"[{this.name}] Committed {new_loc_itr} iters to CPU message.")
        return commit_msg

    def push(this, remote, commit_msg, action="fed_avg_weighed_by_iter"):
        """
        Pushes a pre-constructed commit message to the remote repo.
        """
        if remote is None:
            return

        # Inject the desired remote action into the message
        commit_msg["metadata"]["action"] = action

        remote.push(
            name=this.name,
            state_dict=commit_msg["state_dict"],
            metadata=commit_msg["metadata"]
        )

    def pull(this, remote, action="overwrite"):
        """
        Fetches remote weights and applies the local merge strategy.
        """
        if remote is None:
            return

        # Remote returns: (state_dict, effective_iteration)
        remote_sd, remote_eff_itr = remote.pull(this.name, action=action)

        if remote_sd is not None:
            # Note: Ensure the model is back on the correct device when loading
            # .load_state_dict handles device mapping automatically usually,
            # but we can be explicit if needed.
            this.model.load_state_dict(remote_sd)
            this.effective_iter = remote_eff_itr
            info(f"[{this.name}] Pulled global weights. Global clock: {this.effective_iter}")

    def commit_and_fetch(this, cur_itr, remote_name=None,
                         remote_action="fed_avg_weighed_by_iter",
                         local_action="overwrite"):
        """
        Workflow: Snapshot to CPU -> Push msg -> Pull remote -> Merge locally.
        """
        # 1. Prepare the message (CPU-bound)
        msg = this.commit(cur_itr)

        # 2. Network/Repo Sync
        target_server = remote_name if remote_name is not None else this.server
        if target_server is not None:
            this.push(target_server, msg, remote_action)
            this.pull(target_server, local_action)
    # don't count that on NG

    def debug_show_grad_norm(this):
        """
        Prints the min, max, mean, and standard deviation (std) of gradients
        for each parameter, and counts parameters without a gradient.
        """

        # Initialize counters and totals
        total_params = 0
        params_with_grad = 0
        params_without_grad = 0

        print("---", this.name, "--- Gradient Statistics per Parameter (L2) ---")

        # Iterate over all named parameters in the model
        for name, p in this.model.named_parameters():
            total_params += 1

            # Check if the parameter has a gradient
            if p.grad is not None:
                params_with_grad += 1
                grad = p.grad.data

                # Calculate the L2 norm for the entire gradient tensor of this parameter
                grad_norm = grad.norm(2)

                # Calculate the required statistics directly on the gradient tensor's values
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()

                print(
                    f"✅ [{name}] |Norm|: {grad_norm.item():.4f} | "
                    f"Min: {grad_min:.4f}, Max: {grad_max:.4f}, "
                    f"Mean: {grad_mean:.4f}, Std: {grad_std:.4f}"
                )
            else:
                params_without_grad += 1
                print(f"❌ [{name}] GRADIENT IS NONE (requires_grad=False or detached)")

        print("---------------------------------------------")

        ## 🔢 Gradient Summary
        print(f"Total Parameters: {total_params}")
        print(f"Parameters With Gradient: {params_with_grad}")
        print(f"Parameters **Without Gradient**: {params_without_grad}")


if __name__ == '__main__':
    mod=torch.nn.Conv2d(1,3,4).cuda();
    models = trnp.replicate(mod, ["cuda","cuda"]);
    pass;
