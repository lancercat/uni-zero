import copy

import torch.optim.lr_scheduler

from neko_sdk.CnC.collate.neko_resize_to_fill import neko_resize_and_padfill,neko_resize_and_fill

# we first mimic base behaviour...

from neko_2024_NGNW.common.namescope import mod_names
from neko_sdk.chunked_FE.res45.res45G2 import neko_r45_layers_origNG, neko_r45_norms_origNG
from neko_sdk.chunked_FE.res45.res45g2.res45_g2 import neko_res45_bogo_g2

from neko_sdk.neko_framework_NG.neko_modular_NG import neko_modular_NG as M
from neko_2024_NGNW.common.ak6 import AK6
from neko_2024_NGNW.common.heads_mk3.chaos_head import chaos_head

from neko_sdk.OSR.openset_classifiers.multipart_classifier_kpm_NG import neko_openset_linear_classifierKPM
from neko_sdk.neko_framework_NG.modules.concat_mvn_dev import neko_concat_dev,neko_list_rgb_mvn_dev
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk1 import spatial_attention_NG_mk2
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.bogog2_modules.featmap_to_feat import gen4_featmap_to_feat_abstract
from neko_sdk.neko_framework_NG.modules.neko_label_sampler_NG import neko_prototype_sampler_NG
from osocrNG.ocr_modules_NG.extra_embedding.one_normed_embedding import neko_class_embeddingNG
from neko_sdk.OSR.class_center_mgmt.modules.prototype_gen4 import vis_prototyper_gen4

from osocrNG.ocr_modules_NG.sampler_NG.aggregator import neko_basic_attn_aggr
from neko_sdk.CnC.controls.modules.policy_pred.feature_based_policy_pred import neko_feature_based_static_router
# we try to setup everything but data in one factory.
# I hate 1000 line files but let's keep this way,
# before I figure out what part can be decoupled from this mess.
from osocrNG.ocr_modules_NG.temporal_fe.temporal_fe_fpn import temporal_fe_fpn
from osocrNG.ocr_modules_NG.spatial_embedding.se_dino_like import temporal_se_dino_like_mk1
from neko_2024_NGNW.common.make_deploy_anchor import make_deploy_anchor
from neko_2024_NGNW.common.heads_mk3.head_params import head_common_param

# local length prediction is now default.

# dropout in v5 is still not unified.
# maybe stage this with uniformed fe-sample pipeline after AAAI ddl (v6)
# I need research engineers! adding and checking dropout here and there does not require a PhD degree....
# we don't change modules if possible and all.
class neko_wna_v6_base:
    bn_engine=neko_r45_norms_origNG;
    layers_engine=neko_r45_layers_origNG;
    bogo_fe_engine=neko_res45_bogo_g2;
    temporal_fe_engine=temporal_fe_fpn;
    spatial_embedding_engine=temporal_se_dino_like_mk1;
    expf=1;
    MN=mod_names;
    OPT="adadelta";
    STABLE_PLAN=True;
    BASE_LR = 0.1;
    BASE_decay=0.0005;
    SCHED_override = {
        "engine": torch.optim.lr_scheduler.MultiStepLR,
        "params": {
            "milestones": [3, 5],
            "gamma": 0.1,
        }
    };
    def get_fe_ochs(this,expf):
        return [int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), int(512 * expf),
                                      int(this.feat_ch_model)];

    def set_imgch(this):
        this.img_ch=3;

    def __init__(this,platformcfg:neko_platform_cfg):
        this.seed=9;
        this.platform=platformcfg;
        this.save_path=platformcfg.save_root;
        this.save_each=20000;
        this.set_imgch();

        # Commander
        this.assch=512;
        this.localassch=320;
        this.feat_ch_model=512;

        # padding
        this.margin = ((4, 4), (4, 4));


        # FE
        this.fe_blk_cnt=[None, 3, 4, 6, 6, 3];
        this.fe_ochs= this.get_fe_ochs(this.expf);
        this.fe_strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)];
        this.fe_relu_inplace=False;
        this.fe_bn_affine=False;
        this.fe_cores_registered={};
        this.bns_registered={};
        # TFE
        this.nparts = 1;
        this.num_se_channels=32;
        this.cam_channels=64;


        # proto sampler
        this.capacity=256;
        this.val_frac=0.8;
        this.case_sensitive=False;
        this.head_factory=chaos_head();
    def get_optim_param(this):
        return {
            M.PARAM_save_each: this.save_each,
            M.PARAM_save_path: this.save_path,
            M.PARAM_opt_engine: this.OPT,
            M.PARAM_opt_lr: this.BASE_LR,
            M.PARAM_opt_weight_decay: this.BASE_decay,
            M.PARAM_sched_override: this.SCHED_override,
        }
    def get_transform_optim_param(this):
        return this.get_optim_param();

    #
    #
    # # if you have a factory to handle submodule config.
    # def config_pack(this,modcfgdict,pdict):
    #     for name in pdict["saveable"]:
    #         modcfgdict=this.config_saveable(modcfgdict,pdict["savable"]["type"],pdict["savable"]["param"],name);
    #     for name in pdict["nonsaveable"]:
    #         modcfgdict = this.config_saveable(modcfgdict, pdict["savable"]["type"], pdict["savable"]["param"], name);
    #     return modcfgdict;

    def config_spatial_embedding(this,modcfgdict,name,tags=None):
        e=this.spatial_embedding_engine
        modcfgdict=M.add_config_to_dict(
            modcfgdict,name,e,{
                e.PARAM_scales:  [
                    [this.fe_ochs[1] + this.num_se_channels, 16, 64],
                    [this.fe_ochs[3] + this.num_se_channels, 8, 32],
                    [this.fe_ochs[-1] + this.num_se_channels, 8, 32]
                ],
                e.PARAM_num_se_channels: this.num_se_channels,
            },this.get_optim_param(),tags=tags
        )
        return modcfgdict;

    def config_temporal_fe(this,modcfgdict,name,tags=None):
        e= this.temporal_fe_engine;
        modcfgdict=M.add_config_to_dict(modcfgdict,name, e, {
            e.PARAM_scales: [
                [this.fe_ochs[1] + this.num_se_channels, 16, 64],
                [this.fe_ochs[3] + this.num_se_channels, 8, 32],
                [this.fe_ochs[-1] + this.num_se_channels, 8, 32]
            ],
            e.PARAM_depth:8,
            e.PARAM_num_se_channels:this.num_se_channels,
            e.PARAM_num_channels:this.cam_channels
            }, this.get_optim_param(),tags=tags);
        return modcfgdict


    def get_param_for_head(this,maxT):
        return {
            head_common_param.PARAM_cam_channels:this.cam_channels,
            head_common_param.PARAM_nparts:this.nparts,
            head_common_param.PARAM_maxT:maxT,
            head_common_param.PARAM_localassch:this.localassch,
            head_common_param.PARAM_feat_ch:this.feat_ch_model
        }
    def config_fe_core(this,modcfgdict,name,tags=None):
        modcfgdict = M.add_config_to_dict(modcfgdict,name,this.layers_engine,{
                "inpch": this.img_ch,
                "blkcnt": this.fe_blk_cnt,
                "ochs": this.fe_ochs,
                "strides": this.fe_strides,
                "inplace": this.fe_relu_inplace
            },this.get_optim_param(),tags=tags);
        this.fe_cores_registered[name]=True;
        return modcfgdict;
    def config_dom_bn(this,modcfgdict,name,tags=None):
        modcfgdict = M.add_config_to_dict(modcfgdict,name,this.bn_engine,{
                "strides": this.fe_strides, # Keep the same with fe. And it is actually used to decide how many layers (bns) are in a block.
                "inpch": this.img_ch,
                "blkcnt": this.fe_blk_cnt,
                "ochs": this.fe_ochs,
                "affine": this.fe_bn_affine,
            },this.get_optim_param(),tags=tags);
        this.bns_registered[name]=True;
        return modcfgdict;
    def config_classifier(this,modcfgdict,bogo_dict,prefix,tags=None):
        modcfgdict= M.add_config_to_dict(modcfgdict,prefix+this.MN.WORD_CLASSIFIER,neko_openset_linear_classifierKPM,{},this.get_optim_param(),tags=tags);
        return modcfgdict,bogo_dict;
    def config_prototype_sampling(this,modcfgdict,bogo_dict,prefix,metapath,tags=None):
        modcfgdict= M.add_config_to_dict(modcfgdict,prefix+this.MN.META_SAM,neko_prototype_sampler_NG,{
            "meta_args":
                {
                    "meta_path":metapath, # well we are moving towards a more inclusive meta cfg.
                    "case_sensitive": this.case_sensitive,
                },
            "sampler_args":
                {
                    "max_batch_size": this.capacity,
                    "val_frac": this.val_frac,
                    "neg_servant": True,
                    "seed": this.seed
                },
        },this.get_optim_param(),tags=tags);
        return modcfgdict,bogo_dict
    #
    def config_prototyping(this,modcfgdict,bogo_dict,prefix,tags=None):
        tags=copy.copy(tags);
        e2 = neko_class_embeddingNG; # this won't bite even if you have no ctc parts
        modcfgdict = M.add_config_to_dict(modcfgdict,prefix+this.MN.SP_EMB_MOD,
                                          e2, {
                                              e2.PARAM_channel: this.feat_ch_model,
                                              e2.PARAM_n_parts:this.nparts,
                                              e2.PARAM_transcriptions: [this.MN.SP_TOK_EMP,this.MN.SP_TOK_SEP]
                                          },
                                         this.get_optim_param(),tags);
        modcfgdict,bogo_dict=this.config_fe(modcfgdict,bogo_dict,prefix+this.MN.FE_FE_core_name,prefix+this.MN.PROTO_FE,tags);
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.CHAR_ATT,
            spatial_attention_NG_mk2,
            {spatial_attention_NG_mk2.PARAM_ifc:this.fe_ochs[1] ,
                spatial_attention_NG_mk2.PARAM_nparts: this.nparts,
                spatial_attention_NG_mk2.PARAM_num_se_channels: this.num_se_channels,
            },this.get_optim_param(),tags);
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.PROTO_AGGR,neko_basic_attn_aggr,{},this.get_optim_param(),tags);

        # we are still keeping this for test in v6.
        # after that we dump.

        bogo_dict[prefix + this.MN.PROTO_ENC] = {
            "bogo_mod": vis_prototyper_gen4,
            "args":
                {
                    "mod_cvt":
                        {
                            "backbone": prefix + this.MN.PROTO_FE,
                            "aggr": prefix + this.MN.CHAR_ATT,
                        },
                }
        }
        return modcfgdict,bogo_dict
    def config_lsctc(this,modcfgdict,bogo_dict,prefix):
        e=neko_resize_and_fill;
        modcfgdict = M.add_config_to_dict(modcfgdict,prefix+this.MN.LSCT_CHAR_COLLATE_name,
                e,{
                e.PARAM_target_size:(32,32),
                e.PARAM_interpolate_mode: "bilinear"
                # To have some margins help recognition--- independently mentioned by Chenlvs, Simon
            },None);
        modcfgdict,bogo_dict=this.config_fe(modcfgdict,bogo_dict,prefix+this.MN.FE_FE_core_name,prefix+this.MN.LSCT_CHAR_FE);
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.LSCT_CHAR_ATT,
            spatial_attention_NG_mk2,
            {spatial_attention_NG_mk2.PARAM_ifc:this.fe_ochs[1] ,
                spatial_attention_NG_mk2.PARAM_nparts: this.nparts,
                spatial_attention_NG_mk2.PARAM_num_se_channels: this.num_se_channels,
            },this.get_optim_param());
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.LSCT_CHAR_AGGR,neko_basic_attn_aggr,{},this.get_optim_param());
        return modcfgdict,bogo_dict


    def config_commander_fe(this, modcfgdict, bogo_dict, prefix,tags=None):
        # sets a fe for you router. The fe is not stablized yet, so we won't be sharing components.
        # and this is meant to be a tiny network
        modcfgdict = this.config_fe_core(modcfgdict, prefix + this.MN.ROUTER_FE_core_name,tags);
        modcfgdict, bogo_dict = this.config_fe(modcfgdict, bogo_dict, prefix + this.MN.ROUTER_FE_core_name,
                                               prefix + this.MN.ROUTER_FE_bbn_name,tags);
        modcfgdict = M.add_config_to_dict(modcfgdict,prefix + this.MN.ROUTER_ATT_name,
            spatial_attention_NG_mk2,
            {
                spatial_attention_NG_mk2.PARAM_ifc: this.fe_ochs[1],
                spatial_attention_NG_mk2.PARAM_nparts: this.nparts,
             },
        this.get_optim_param(),tags
        );

        bogo_dict[prefix + this.MN.ROUTER_AGGR_name] = {
            "bogo_mod": gen4_featmap_to_feat_abstract,
            "args":
                {
                    gen4_featmap_to_feat_abstract.PARAM_MODCVT:
                        {
                            "aggr": prefix + this.MN.ROUTER_ATT_name,
                        },
                }
        }
        return modcfgdict, bogo_dict;


    def config_commander_planmaker(this,modcfgdict,prefix,module_anchor_config):
        e=neko_feature_based_static_router;
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.ROUTER_CMD_name,e,
                                        {e.PARAM_stablize_with_norm:this.STABLE_PLAN,e.PARAM_indim:this.feat_ch_model,e.PARAM_expert_names:module_anchor_config["names"]},
                                         this.get_optim_param());
        return modcfgdict;


    # v6 does not have global length prediction by default
    def config_global_task_mods(this,modcfgdict,bogo_dict,prefix,anchor_dict):
        return modcfgdict,bogo_dict;

    def config_commander_core(this,modcfgdict,bogo_dict,prefix,anchor_dict,tags=None):
        modcfgdict,bogo_dict= this.config_commander_fe(modcfgdict, bogo_dict, prefix);
        modcfgdict,bogo_dict=this.config_global_task_mods(modcfgdict,bogo_dict,prefix,anchor_dict);
        modcfgdict=this.config_commander_planmaker(modcfgdict,prefix,anchor_dict);
        return modcfgdict,bogo_dict;


    def config_dom_bogofe(this,bogocfg_dict,name,conv_container,bn_container):
        bogocfg_dict[name]={
            "bogo_mod": this.bogo_fe_engine,
            "args":
                {
                    "mod_cvt":
                        {
                            "conv": conv_container,
                            "norm": bn_container,
                        },
                }
        }
        return bogocfg_dict;


    def config_fe(this,modcfgdict,bogo_modcfgdict,layers_name,name,tags=None):
        assert layers_name in this.fe_cores_registered;
        if(name+"_bn") not in this.bns_registered:
            modcfgdict=this.config_dom_bn(modcfgdict,name+"_bn",tags=tags);
        bogo_modcfgdict=this.config_dom_bogofe(bogo_modcfgdict,name,layers_name,name+"_bn");
        return modcfgdict,bogo_modcfgdict;




    def config_mvn_mods(this, modcfgdict,prefix,tags=None):
        return M.add_config_to_dict(modcfgdict,prefix+this.MN.MVN_name,neko_concat_dev,{
                "mean":[127.5],
                "var":[128],
            },None,tags);


    def config_collate_mods(this,modcfgdict,bogo_dict,prefix,module_anchor_config,placementcfg_dict=None):
        target_size_wh=module_anchor_config[AK6.target_size_wh];
        if(AK6.padding_size in module_anchor_config):
            padsize=module_anchor_config[AK6.padding_size];
        else:
            padsize=this.margin;
        rtsize=[
            target_size_wh[1]+sum(this.margin[1]),
            target_size_wh[0] + sum(this.margin[0])
        ];
        e=neko_resize_and_padfill;
        modcfgdict = M.add_config_to_dict(modcfgdict,prefix+this.MN.COLLATE_name,
                e,{
                e.PARAM_target_size:rtsize,
                e.PARAM_padval:0,
                e.PARAM_margins:padsize,
                e.PARAM_interpolate_mode: "bilinear"
                # To have some margins help recognition--- independently mentioned by Chenlvs, Simon
            },None);
        return modcfgdict,bogo_dict;


    def config_shared_facility(this,modcfgdict,bogo_dict,prefix,module_anchor_config,tags=None):
        modcfgdict= this.config_mvn_mods(modcfgdict,prefix,tags);
        modcfgdict,bogo_dict=this.config_commander_core(modcfgdict,bogo_dict,prefix,module_anchor_config);
        modcfgdict = this.config_fe_core(modcfgdict, prefix+this.MN.FE_FE_core_name);
        modcfgdict,bogo_dict=this.config_prototyping(modcfgdict,bogo_dict,prefix);
        modcfgdict,bogo_dict=this.config_classifier(modcfgdict,bogo_dict,prefix);
        return modcfgdict,bogo_dict;
    # Module anchors does not have to be the same with data anchors.
    # The mapping is learnt from the data with rewards.

    def config_expert(this,modcfgdict,bogo_dict,prefix,module_anchor_config,a):
        # mk6 still does not support device affinity.

        modcfgdict, bogo_dict = this.config_collate_mods(modcfgdict, bogo_dict, prefix + a, module_anchor_config[a])

        modcfgdict, bogo_dict = this.config_fe(modcfgdict, bogo_dict, prefix+ this.MN.FE_FE_core_name,
                                               prefix +a + this.MN.WORD_FE);
        # the anchor now need to report if it needs a tfe and if needs
        if(module_anchor_config[a][AK6.has_tfe]):
            modcfgdict = this.config_spatial_embedding(modcfgdict, prefix +a + this.MN.WORD_TEMPORAL_SE);
            modcfgdict = this.config_temporal_fe(modcfgdict, prefix+a + this.MN.WORD_TEMPORAL_FE);
        for head in module_anchor_config[a][AK6.heads][AK6.head_names]:
            modcfgdict=this.head_factory.get_head_mod_by_string(modcfgdict,prefix+a,head, module_anchor_config[a][AK6.heads][head]["type"],this.get_param_for_head(module_anchor_config[a][AK6.maxT]),this.get_optim_param());
        return modcfgdict,bogo_dict;
    def config_core_modules(this,modcfgdict,bogo_dict, module_anchor_config, prefix=""):
        # you can merge two loops, its not like order matters but let's keep it more readable.

        modcfgdict, bogo_dict = this.config_shared_facility(modcfgdict, bogo_dict, prefix, module_anchor_config);
        for a in module_anchor_config[AK6.names]:
            modcfgdict,bogo_dict=this.config_expert(modcfgdict,bogo_dict,prefix,module_anchor_config,a);

       # V5 got rid of all dropouts since they don't help much.
       # modcfgdict=this.config_drop_mods(modcfgdict,prefix);
        return modcfgdict, bogo_dict;

    # the reward used in this version is non-paramatic and is handled by an agent.
    # you can override this if you want to do something more complex.
    def config_routing_reward(this,modcfgdict,bogo_dict, module_anchor_config, prefix):
        return modcfgdict,bogo_dict






    def config_training_extra(this, modcfgdict, bogo_dict, module_anchor_config, metapath, prefix=""):
        modcfgdict, bogo_dict = this.config_prototype_sampling(modcfgdict, bogo_dict, prefix, metapath);
        modcfgdict, bogo_dict = this.config_routing_reward(modcfgdict, bogo_dict, module_anchor_config, prefix);
        for a in module_anchor_config[AK6.names]:
            for h in module_anchor_config[a][AK6.heads][AK6.head_names]:
                modcfgdict=this.head_factory.get_head_training_extra_mod_by_string(modcfgdict,prefix+a,h,module_anchor_config[a][AK6.heads][h]["type"],this.get_param_for_head(module_anchor_config[a][AK6.maxT]),this.get_optim_param());
        # if the branch has a local len prediction.

        return modcfgdict, bogo_dict;

    # set up modules shared by heads

    # deploy replication for some anchors.

    # now the databalancer is decoupled with anchors, so no need to meddle with it
    def config_for_testing(this,modcfgdict,bogo_dict, module_anchor_config, prefix=""):
        return this.config_core_modules(modcfgdict,bogo_dict, module_anchor_config, prefix);



    # why having global prefix you ask?
    # Well if you want to implement FL via project hanazo then you will find them useful :-)
    # now models can also have tags for grouped control. Consider use them when implementing FL.
    def config_for_training(this,modcfgdict,bogo_dict, module_anchor_config,metapath, prefix=""):
        actual_deploy_anchor_config=make_deploy_anchor(module_anchor_config);
        modcfgdict,bogo_dict= this.config_core_modules(modcfgdict,bogo_dict,actual_deploy_anchor_config,prefix);
        return this.config_training_extra(modcfgdict,bogo_dict,actual_deploy_anchor_config,metapath,prefix);



class neko_wna_v6_base_no_pad(neko_wna_v6_base):
    def config_collate_mods(this,modcfgdict,bogo_dict,prefix,module_anchor_config,placementcfg_dict=None):
        target_size_wh=module_anchor_config[AK6.target_size_wh];
        e=neko_resize_and_fill;
        modcfgdict = M.add_config_to_dict(modcfgdict,prefix+this.MN.COLLATE_name,
                e,{
                e.PARAM_target_size:(target_size_wh[1],target_size_wh[0]),
                e.PARAM_interpolate_mode: "bilinear"
                # To have some margins help recognition--- independently mentioned by Chenlvs, Simon
            },None);
        return modcfgdict,bogo_dict;
class neko_wna_v6XL_base_no_pad(neko_wna_v6_base_no_pad):
    expf = 1.5;
class neko_wna_v6XL_lang_base_no_pad(neko_wna_v6_base_no_pad):
    expf = 1.5;
    def config_for_testing(this,modcfgdict,bogo_dict, module_anchor_config, prefix=""):
        return this.config_core_modules(modcfgdict,bogo_dict, module_anchor_config, prefix);

    # why having global prefix you ask?
    # Well if you want to implement FL via project hanazo then you will find them useful :-)
    def config_for_training(this,modcfgdict,bogo_dict, module_anchor_config,metapath, prefix=""):
        actual_deploy_anchor_config=make_deploy_anchor(module_anchor_config);
        modcfgdict,bogo_dict= this.config_core_modules(modcfgdict,bogo_dict,actual_deploy_anchor_config,prefix);
        return this.config_training_extra(modcfgdict,bogo_dict,actual_deploy_anchor_config,metapath,prefix);
    def config_for_testing(this,modcfgdict,bogo_dict, module_anchor_config, prefix=""):
        return this.config_core_modules(modcfgdict,bogo_dict, module_anchor_config, prefix);


class neko_wna_v6XXL_base_no_pad(neko_wna_v6_base_no_pad):
    expf = 2;

class neko_wna_v6_base_no_pad_lfelr03(neko_wna_v6_base_no_pad):
    def config_shared_facility(this, modcfgdict, bogo_dict, prefix, module_anchor_config, tags=None):
        modcfgdict= this.config_mvn_mods(modcfgdict,prefix);
        modcfgdict,bogo_dict=this.config_commander_core(modcfgdict,bogo_dict,prefix,module_anchor_config);
        modcfgdict = this.config_fe_core(modcfgdict, prefix+this.MN.FE_FE_core_name);
        modcfgdict[prefix+this.MN.FE_FE_core_name]["learning_rate"]*=0.3;
        modcfgdict,bogo_dict=this.config_prototyping(modcfgdict,bogo_dict,prefix);
        modcfgdict,bogo_dict=this.config_classifier(modcfgdict,bogo_dict,prefix);
        return modcfgdict,bogo_dict;
