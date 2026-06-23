import copy

from neko_sdk.MJT.common import Updata_Parameters
from neko_sdk.MJT.utils import update
from neko_sdk.neko_framework_NG.neko_modular_NG import neko_modular_NG
from neko_sdk.thirdparty.mmdetapply import multi_apply
from neko_sdk.log import warn,info,fatal




class neko_module_opt_setNG:

    def get_modular_dict(this,key="NEP_main_NEP"):
        return get_modular_dict(this,key);

    # well if you insist on data paralleling, fork and put them into moddict
    def get_maybe_replicated_real_mods(this,key):
        rmd={};
        for k in this.real_modulars:
            rmd[k]=this.real_modulars[k].get_mod(key); # if the model does not have a replica, return the main model.
        return rmd;

    def add_replica_grp(this,name,dev):
        assert (name not in this.replica_groups);
        this.replica_groups[name]=dev;
    def arm_replica_grps(this,dic):
        for i in dic:
            this.add_replica_grp(i,dic[i]);
    def duplicate(this,grp_key,mod_key,device_overide=None):
        assert (grp_key in this.replica_groups);



    def clear(this):
        this.optimizers = [];
        this.optnames = [];
        this.optimizer_schedulers = [];

        this.real_modulars = {};
        this.mods={};
        this.replica_groups={};
        this.device_mod_dict={};

        this.bogo_modular_list = [];
        this.bogo_config_dict = {};
        this.bogo_mappings = {};

        # will stage by 320

    def register_bogo(this,name,bogocfg):
        this.bogo_modular_list.append(name);
        this.bogo_config_dict[name]=bogocfg;
    def register_real(this,name,mod):
        this.real_modulars[name] = mod;
        if (this.real_modulars[name].optimizer is not None):
            this.optnames.append(name);
            this.optimizers.append(this.real_modulars[name].optimizer);
            this.optimizer_schedulers.append(this.real_modulars[name].optimizer_sched);

    def arm_modules(this, real_modular_cfg:dict[neko_modular_NG], bogo_mod_cfg,decay_override=None):
        this.clear();
        # we keep them to make replications;
        for name in bogo_mod_cfg:
            this.register_bogo(name,bogo_mod_cfg[name])
        for name in real_modular_cfg:
            if(decay_override):
                real_modular_cfg[name]["weight_decay"]=decay_override;
            this.register_real(name,neko_modular_NG.get_default_NG_modular(real_modular_cfg[name]));

    def train_mode(this,tag=None):
        for modk in this.real_modulars:
            if ((tag is not None) and
                (tag not in this.real_modulars[modk].tags)):
                continue;
            this.real_modulars[modk].train();
    def eval_mode(this,tag=None):
        for modk in this.real_modulars:
            if((tag is not None) and
               (tag not in this.real_modulars[modk].tags)):
                continue;
            this.real_modulars[modk].eval();
    def to(this,device,tag=None):
        for modk in this.real_modulars:
            if((tag is not None) and
               (tag not in this.real_modulars[modk].tags)):
                continue;
            this.real_modulars[modk].to(device);
    def set_lr(this,lr,tag=None):
        for modk in this.real_modulars:
            if((tag is not None) and
               (tag not in this.real_modulars[modk].tags)):
                continue;
            if(this.real_modulars[modk].optimizer is not None):
                for g in this.real_modulars[modk].optimizer.param_groups:
                    g['lr'] =lr;

    def place(this,device_map,tag=None):
        for modk in this.real_modulars:
            if ((tag is not None) and
                    (tag not in this.real_modulars[modk].tags)):
                continue;
            this.real_modulars[modk].place(device_map[modk],this.replica_groups);
    def bfloat16(this,tag=None):
        for modk in this.real_modulars:
            if ((tag is not None) and
                    (tag not in this.real_modulars[modk].tags)):
                continue;
            this.real_modulars[modk].bfloat16();

    def distribute(this,devices):
        pass; # well if you want to distribute the model on several devices here would be your chance.

    def save_necessary(this,nEpoch, batch_idx):
        for modk in this.real_modulars:
            this.real_modulars[modk].save_if_needed(nEpoch, batch_idx);
    def load(this,itrkey):
        for modk in this.real_modulars:
            this.real_modulars[modk].load(modk,itrkey);
    def update_para(this):
        multi_apply(update, this.optimizers);

    def update(this):
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            fatal("error");
    def update_opt(this,epoch_idx):
        for opts in this.optimizer_schedulers:
            opts.step(epoch_idx);
    def zero_grad(this):
        for opt in this.optimizers:
            opt.zero_grad();
    def norm_grad(this):
        for modk in this.real_modulars:
            if (this.real_modulars[modk].save_each > 0):
                this.real_modulars[modk].normgrad();
# this module connects the weights to a remote repo, which is shared by different instances running different data.
#
# from neko_sdk.hanazo.libhanazo.commit import commit
# from neko_sdk.hanazo.libhanazo.flowerbed import neko_flowerbed
# # project hanazo is not forgotten.... It's just I have too much at my paws.
#
# class neko_module_opt_setNG_hanazo(neko_module_opt_setNG):
#     pass;


def attempt_arm_bogo_list( bogolist, modcfgs, modular_dict):
    fail_list = [];
    for name in bogolist:
        cfg = modcfgs[name];
        # bogo modules are re-combination of parts of existing real and bogo modules.
        try:
            mod = cfg["bogo_mod"](cfg["args"], modular_dict);
            modular_dict[name] = mod;
        except:
            fail_list.append(name);
    return modular_dict, fail_list;

def get_modular_dict(modset,key="NEP_main_NEP"):
    modular_dict = {};
    rmd=modset.get_maybe_replicated_real_mods(key);
    for k in rmd:
        modular_dict[k] = rmd[k];

    list_bogo_to_arm = copy.copy(modset.bogo_modular_list);
    for i in range(40):
        if (len(list_bogo_to_arm) == 0):
            break;
        if (i):
            print("Attempt", i, "for", list_bogo_to_arm);
        modular_dict, list_bogo_to_arm = attempt_arm_bogo_list(list_bogo_to_arm, modset.bogo_config_dict,
                                                                    modular_dict);
    if (len(list_bogo_to_arm)):
        print("failed dependency for module(s):", list_bogo_to_arm, "please check dependency");
        fatal("error");
    return modular_dict;

