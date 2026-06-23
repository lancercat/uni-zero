from neko_sdk.neko_framework_NG.neko_modular_NG import neko_modular_NG as M
import torch
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_sdk.neko_framework_NG.names import P

from typing import List, Optional, Dict, Any

class virtual_factory:
    MOD_PRFX_local="local_mod_prfx";

    # we need agtprfx as agt can be an outer product of data and mod, and it can be smth else. but let's put it this way.

class global_mod_cfg:
    PARAM_save_each = "save_each";
    PARAM_DECAY="decay";
    PARAM_expf = "expf";
    PARAM_OPT="opt";
    DFT_OPT = "adadelta";
    BASE_LR = 0.1;
    BASE_decay = 0.0005;
    PARAM_BN_HAS_AFFINE= "bn_has_affine";
    PARAM_FEAT_DIM="feat_dim";
    MAX_grad_norm=5;
    # MN=MN;
    # AN = AN;
    # VN=VN; # well let's do this here
    SCHED_override = {
        "engine": torch.optim.lr_scheduler.MultiStepLR,
        "params": {
            "milestones": [3, 5],
            "gamma": 0.1,
        }
    };

    def __init__(this, platformcfg: neko_platform_cfg, params):
        this.seed = 9;
        this.save_each = 20000;
        this.platform = platformcfg;
        this.save_path = platformcfg.save_root;
        this.expf = neko_get_arg(this.PARAM_expf, params, 1);
        this.opt=neko_get_arg(this.PARAM_OPT,params,this.DFT_OPT);
        this.tfe_chs=128; # channels for temporal features.
        # endpoint channels
        this.img_ch = 3;
        this.assch = 512;
        this.feat_ch_model = neko_get_arg(this.PARAM_FEAT_DIM,params,512);
        this.decay=neko_get_arg(this.PARAM_DECAY,params,this.BASE_decay);
        this.num_se_channels = 64;
        this.fe_ochs = [int(32 * this.expf), int(64 * this.expf), int(128 * this.expf), int(256 * this.expf),
                        int(512 * this.expf),
                        int(this.feat_ch_model)];
        this.nparts = 1;
        this.bn_affine=neko_get_arg(this.PARAM_BN_HAS_AFFINE,params,False);
    pass;


class virtual_mod_factory(virtual_factory):
    KEY_global_registry="global_registry";
    KEY_modules="modules";
    KEY_bogo_moduels="bogos";
    @classmethod
    def empty_real_modcfg(cls):
        return {
            cls.KEY_modules:{},
            cls.KEY_global_registry:{}
        };
    @classmethod
    def empty_bogomod_cfgs(cls):
        return {}; # these are supposed to be tightly tied modules that are too complex to be used by agents.
                        # but heck we discourage using this, and may be phased out in early 33x.
    @classmethod
    def empty_modcfgs(cls):
        return cls.empty_real_modcfg(), cls.empty_bogomod_cfgs();

    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        pass;



    def get_optim_param(this):
        return {
            M.PARAM_save_each: this.gmodcfg.save_each,
            M.PARAM_save_path: this.gmodcfg.save_path,
            M.PARAM_opt_engine: this.gmodcfg.opt,
            M.PARAM_opt_lr: this.gmodcfg.BASE_LR,
            M.PARAM_max_grad_norm: this.gmodcfg.MAX_grad_norm,
            M.PARAM_opt_weight_decay: this.gmodcfg.decay,
            M.PARAM_sched_override: this.gmodcfg.SCHED_override,
        }
    def config_non_saveable(this,modcfgdict,mod_type,mod_param,name,tags=None):
        modcfgdict[this.KEY_modules][name] = {
             M.PARAM_save_each: 0,
             M.PARAM_save_path: "NEP_skipped_NEP",
             M.PARAM_name: name, # this sets an internal name.
             M.PARAM_mod_type: mod_type,
             M.PARAM_mod_param: mod_param,
            M.PARAM_tags:tags
        }
        return modcfgdict;
    def config_saveable(this,modcfgdict,mod_type,mod_param,name,tags=None):
        cfg= this.get_optim_param();
        cfg[M.PARAM_name]=name;
        cfg[M.PARAM_mod_type]=mod_type;
        cfg[M.PARAM_mod_param]=mod_param;
        cfg[M.PARAM_tags]=tags;
        if(name in modcfgdict[this.KEY_modules]):
            fatal("????");
        modcfgdict[this.KEY_modules][name]=cfg;
        return modcfgdict;

    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict:Dict[str,Any]=None):
        this.gmodcfg=gmodcfg;
        this.external_factory_dict=external_factory_dict;


    pass;
class virtual_agt_factory (virtual_factory):
    DATA_PRFX_local="local_data_prfx";
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        pass;


# this is basically
class virtual_part_factory(virtual_mod_factory):
    DATA_PRFX_local="local_data_prfx";
    def __init__(this, platform_cfg: neko_platform_cfg,gmodcfg:global_mod_cfg, params):
        super().__init__(gmodcfg,params);
        this.platform_cfg = platform_cfg;
        this.setup_factories();
    def setup_factories(this):
        pass;

