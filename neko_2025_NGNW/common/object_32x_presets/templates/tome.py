# datta
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile,neko_im_meta_template,neko_sideinfo_repo
from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
# meta
from neko_2025_NGNW.common.object_32x_presets.templates.data_processing_profile import neko_datastream_processing_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_processing_profile import neko_metastream_processing
from neko_2025_NGNW.common.object_32x_presets.templates.task_processing_profile import neko_task_processing
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.log import fatal
from neko_sdk.neko_framework_NG.names import P,PLST

from dataclasses import dataclass, field,asdict
from typing import Dict, Any, List,Tuple

# building plans: datatree, model tree and task tree module bindings,etc
# some parts are nerfed--- please wait till we find a proper slot to publish the full 32x framework
class neko_320_net_size_meta:
    DFT_ITEM_IM_DATA_SIZE: Tuple[int, int] = [32,32];
    DFT_WORD_IM_DATA_SIZE: Tuple[int,int] = [32,128];
    DFT_WORLD_IM_DATA_SIZE: Tuple[int,int ]=[256,256]
    #
    DFT_SEQ_IM_MOE_SIZES: Dict[str, List[int]] =  {
            "hori": [32, 128],
            "rect": [64, 64],
            "vert": [128, 32],
        };
    PARAM_ITEM_IM_DATA_SIZE="item_data_size";
    PARAM_WORD_IM_DATA_SIZE = "seq_data_size";
    PARAM_WORLD_IM_DATA_SIZE = "world_data_size"; # seikai no wa ru i.
    PARAM_SEQ_IM_MOE_SIZES = "seq_data_size_moe";

    def __init__(this,param):
        this.item_data_size=neko_get_arg(this.PARAM_ITEM_IM_DATA_SIZE,param,this.DFT_ITEM_IM_DATA_SIZE);
        this.word_im_data_size=neko_get_arg( this.PARAM_WORD_IM_DATA_SIZE,param, this.DFT_WORD_IM_DATA_SIZE);
        this.world_im_data_size=neko_get_arg( this.PARAM_WORLD_IM_DATA_SIZE,param, this.DFT_WORLD_IM_DATA_SIZE);
        this.seq_im_moe_size=neko_get_arg(this.PARAM_SEQ_IM_MOE_SIZES,param,  this.DFT_SEQ_IM_MOE_SIZES);




# bcs testing task will need not sharing or whatsoever, thus it can be simple like this
# and also note so far we only support testing it with one data, one meta, and one task.
# in the future we will eventual revise this.
@dataclass
class neko_simple_320_test_task_tome:
    data_ldr_profile: Dict[str, Any] = field(default_factory=dict);
    meta_ldr_profile: Dict[str, Any] = field(default_factory=dict);

    data_process_profile: Dict[str, Any] = field(default_factory=dict);
    meta_process_profile: Dict[str,Any]=field(default_factory=dict);
    task_process_profile: Dict[str,Any]=field(default_factory=dict);
    task_mod_name: str = field(default_factory=str),

    task_disable_unk: bool=False,

    disk_ldr_acfg: Dict[str, Any] = field(default_factory=dict);
    meta_ldr_acfg: Dict[str, Any] = field(default_factory=dict);
    sinfo_ldr_acfg: Dict[str, Any]= field(default_factory=dict);

# bp when possible to release vram
@dataclass
class neko_320_fpbpgrp_tome:

    feat_eps: Dict[str, Any] = field(default_factory=dict); # for feature maps
    item_roi_eps: Dict[str, Any] = field(default_factory=dict); # for individual item classification
    # set_roi_eps: Dict[str,Any]=field(default_factory=dict); # for object detections
    seq_roi_eps: Dict[str, Any] = field(default_factory=dict); # for word/line recogntion

    data_item_im_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);
    data_set_im_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);
    data_seq_im_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);
    # data_panoptic_im_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);


    im_meta_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);
    longid_meta_stream_handler_profiles: Dict[str, Any] = field(default_factory=dict);

    meta_eps: Dict[str, Any] = field(default_factory=dict);

    imcls_task_handlers: Dict[str, Any] = field(default_factory=dict);
    ordered_roiseq_task_handlers: Dict[str, Any] = field(default_factory=dict);
    # semseg_task_handlers: Dict[str,Any]= field(default_factory=dict);
    # insseg_task_handlers: Dict[str,Any]= field(default_factory=dict);
    # xtcs_task_handlers: Dict[str, Any] = field(default_factory=dict);


    def mount_datastream_im_item_core(this,name,dldr_ep,modprfx,size):
        pcfg=neko_datastream_processing_profile.get_default_im_process(
                dldr_ep,modprfx,
                size);
        this.data_item_im_stream_handler_profiles[dldr_ep] = pcfg;
        if (name not in this.item_roi_eps):
            this.item_roi_eps[name] = [];
        if (name not in this.feat_eps):
            this.feat_eps[name] = [];
        this.item_roi_eps[name] += pcfg[neko_datastream_processing_profile.ROI_EPS];
        this.feat_eps[name] += pcfg[neko_datastream_processing_profile.FEAT_EPS];

    def mount_datastream_word_im_seq_core(this,name,dldr_ep,modprfx,size):
        pcfg=neko_datastream_processing_profile.get_default_word_process(
                dldr_ep,modprfx,
                size);
        this.data_seq_im_stream_handler_profiles[dldr_ep] = pcfg;
        if (name not in this.seq_roi_eps):
            this.seq_roi_eps[name] = [];
        if (name not in this.feat_eps):
            this.feat_eps[name] = [];

        this.seq_roi_eps[name] += pcfg[neko_datastream_processing_profile.ROI_EPS];
        this.feat_eps[name] += pcfg[neko_datastream_processing_profile.FEAT_EPS];


    def mount_metasream_im_item_core(this,name,dldr_ep,modprfx,size,tie_mod=False):
        pcfg = neko_metastream_processing.get_default_im_meta_process(dldr_ep, 128, size,
                                                                      modprfx,mod_tied=tie_mod);
        this.im_meta_stream_handler_profiles[dldr_ep] = pcfg;
        if (name not in this.meta_eps):
            this.meta_eps[name] = [];
        this.meta_eps[name] += pcfg[
            neko_metastream_processing.MODSCP_ENDPOINTS];  # modscps are armed and ready sideinfo

# during training, several tasks can share data streams, meta streams, or both with other tasks, so we have to manage them separately
@dataclass
class neko_320_tome(neko_320_fpbpgrp_tome):
    def to_dict(this):
        return {k: v for k, v in asdict(this).items()};

    # Convert __init__ attributes to dataclass fields with mutable defaults handled by default_factory
    diskldr_dprofile_dict: Dict[str, Any] = field(default_factory=dict);
    diskldr_mprofile_dict: Dict[str, Any] = field(default_factory=dict);
    diskldr_lprofile_dict: Dict[str, Any] = field(default_factory=dict);

    # will need to wrap this with taskgrps later
    # gist is that if you just share the modules not the data, per task bp helps with vram utilization
    fpbpgrps : Dict[str, neko_320_fpbpgrp_tome] = field(default_factory=dict);

    imcls_testing_tasks: Dict[str, neko_simple_320_test_task_tome] = field(default_factory=dict);
    im_seqcls_testing_tasks: Dict[str, neko_simple_320_test_task_tome] = field(default_factory=dict);
    net_size_meta: neko_320_net_size_meta =neko_320_net_size_meta({});
    # The original methods below keep 'this' as the instance variable name
    # to maintain the original API signature.
    save_each: int=20000;

    # here we still have to tell which type of datastream--- this could have been put into cfg, but will be done in 33x
    def mount_datastream_im_item(this,name,data_profile,fpbpgrp="main"):
        this.diskldr_dprofile_dict[name] = data_profile;
        if (fpbpgrp not in this.fpbpgrps):
            this.fpbpgrps[fpbpgrp] = neko_320_fpbpgrp_tome();

        for dldr_ep in data_profile[neko_datastream_profile.ENDPOINTS]:
            this.mount_datastream_im_item_core(name,dldr_ep,data_profile[neko_datastream_profile.SIGNATURES][dldr_ep],this.net_size_meta.item_data_size);
            this.fpbpgrps[fpbpgrp].mount_datastream_im_item_core(name,dldr_ep,data_profile[neko_datastream_profile.SIGNATURES][dldr_ep],this.net_size_meta.item_data_size);

    # panoptic will mux this,
    # we will have hd attention (XA) by default
    # for sem mask we will use differnet net config--- not messing it around here.
    # for now--- forget the 'shall all layers' stuff.

    def mount_abstract_datastream_im_seq(this,name,data_profile,size,fpbpgrp):
        this.diskldr_dprofile_dict[name] = data_profile;
        if (fpbpgrp not in this.fpbpgrps):
            this.fpbpgrps[fpbpgrp] = neko_320_fpbpgrp_tome();
        for dldr_ep in data_profile[neko_datastream_profile.ENDPOINTS]:
            this.mount_datastream_word_im_seq_core(name,dldr_ep,data_profile[neko_datastream_profile.SIGNATURES][dldr_ep],size);
            this.fpbpgrps[fpbpgrp].mount_datastream_word_im_seq_core(name,dldr_ep,data_profile[neko_datastream_profile.SIGNATURES][dldr_ep],size);

    def mount_word_datastream_im_seq(this,name,data_profile,fpbpgrp="main"):
        return this.mount_abstract_datastream_im_seq(name,data_profile,this.net_size_meta.word_im_data_size,fpbpgrp);

    def mount_im_meta_stream(this,name,meta_profile,fpbpgrp="main"):
        this.diskldr_mprofile_dict[name]=meta_profile;
        if (fpbpgrp not in this.fpbpgrps):
            this.fpbpgrps[fpbpgrp] = neko_320_fpbpgrp_tome();
        for dldr_ep in meta_profile[neko_meta_stream_profile.SIDE_INFO_REPOS]:
            this.mount_metasream_im_item_core(name,dldr_ep, meta_profile[neko_meta_stream_profile.SIDE_INFO_MBND][dldr_ep],this.net_size_meta.item_data_size,tie_mod=meta_profile[neko_meta_stream_profile.FE_MOD_TIED]);
            this.fpbpgrps[fpbpgrp].mount_metasream_im_item_core(
                name,dldr_ep,
                meta_profile[neko_meta_stream_profile.SIDE_INFO_MBND][dldr_ep],
                this.net_size_meta.item_data_size);


    def add_simple_oscr_task(this,data_ep_name,meta_ep_name,task_mod_name,task_ep_name=None,fpbpgrp="main"):
        if (fpbpgrp not in this.fpbpgrps):
            this.fpbpgrps[fpbpgrp] = neko_320_fpbpgrp_tome();
        if(task_ep_name is None):
            task_ep_name=PLST(["ITEM",data_ep_name,meta_ep_name]);
        this.imcls_task_handlers[task_ep_name]=neko_task_processing.get_default_single_item_oscr_training_task(
            data_ep_name,meta_ep_name,task_ep_name,task_mod_name);
        this.fpbpgrps[fpbpgrp].imcls_task_handlers[task_ep_name] = neko_task_processing.get_default_single_item_oscr_training_task(
            data_ep_name, meta_ep_name, task_ep_name, task_mod_name);

    # this parasite feature maps from other task
    def add_simple_osseq_task(this, data_ep_name, meta_ep_name, task_mod_name, task_ep_name=None, fpbpgrp="main",maxT=32):
        if (fpbpgrp not in this.fpbpgrps):
            this.fpbpgrps[fpbpgrp] = neko_320_fpbpgrp_tome();
        if (task_ep_name is None):
            task_ep_name = PLST(["SEQ", data_ep_name, meta_ep_name]);
        this.ordered_roiseq_task_handlers[task_ep_name] = neko_task_processing.get_default_seq_training_task(
            data_ep_name, meta_ep_name, task_ep_name, task_mod_name,maxT);
        this.fpbpgrps[fpbpgrp].ordered_roiseq_task_handlers[
            task_ep_name] = neko_task_processing.get_default_seq_training_task(
            data_ep_name, meta_ep_name, task_ep_name, task_mod_name,maxT);
        pass;




    def arm_testing_data(this,task_tome:neko_simple_320_test_task_tome,data_ldr_acfg,meta_ldr_acfg,sinfo_ldr_acfg):
        task_tome.disk_ldr_acfg=data_ldr_acfg;
        task_tome.meta_ldr_acfg = meta_ldr_acfg;
        task_tome.sinfo_ldr_acfg = sinfo_ldr_acfg;

    def mount_testing_char_osr_im(this,name,test_data_profile,testing_meta_profile,task_mod_prfx,disable_unk):
        this.imcls_testing_tasks[name]=neko_simple_320_test_task_tome();
        this.imcls_testing_tasks[name].data_ldr_profile={name:test_data_profile};
        this.imcls_testing_tasks[name].meta_ldr_profile={name:testing_meta_profile};
        this.imcls_testing_tasks[name].task_disable_unk=disable_unk;
        deps =test_data_profile[neko_datastream_profile.ENDPOINTS];

        # don't make more than one data stream for testing unless you are up to some testing time per batch  referenced calibration.
        # but that will be far beyond the scope of 32x.
        # srsly i can think of a few use cases--- but hu cares except very niche ocr ppl?
        assert (len(deps)==1);
        dldr_ep=deps[0];
        data_mod_prfx=test_data_profile[neko_datastream_profile.SIGNATURES][dldr_ep];
        # note mod configs are trees--
        # basic sharing controls are done through global shared models
        # Advanced sharing controls are done through bogomods
        dpcfg= neko_datastream_processing_profile.get_default_im_process(dldr_ep,data_mod_prfx, this.net_size_meta.item_data_size);
        this.imcls_testing_tasks[name].data_process_profile = dpcfg;

        meps = testing_meta_profile[neko_meta_stream_profile.ENDPOINTS];

        # well you may want to ensemble/choose from different metas, but that aint happen in 32x
        assert (len(meps)==1);
        mldr_ep =meps[0];
        msig=testing_meta_profile[neko_meta_stream_profile.SIDE_INFO_MBND][mldr_ep];
        mpcfg= neko_metastream_processing.get_default_im_meta_process(mldr_ep, 128,this.net_size_meta.item_data_size,msig);
        this.imcls_testing_tasks[name].meta_process_profile = mpcfg;

        roi_eps= dpcfg[neko_datastream_processing_profile.ROI_EPS];
        assert (len(roi_eps)==1); # this config does not allow routing
        ap_eps=mpcfg[neko_metastream_processing.MODSCP_ENDPOINTS];
        assert (len(ap_eps)==1); # this config does not allow meta selection, routing, or ensembling

        this.imcls_testing_tasks[name].task_mod_name =task_mod_prfx;
        this.imcls_testing_tasks[name].task_process_profile[name]=neko_task_processing.get_default_single_char_oscr_testing_task(
            roi_eps[0], ap_eps[0], name,task_mod_prfx);


        pass;
    def mount_testing_word_osr_im(this,name,test_data_profile,testing_meta_profile,task_mod_prfx,disable_unk):
        this.im_seqcls_testing_tasks[name]=neko_simple_320_test_task_tome();
        this.im_seqcls_testing_tasks[name].task_disable_unk=disable_unk;
        this.im_seqcls_testing_tasks[name].data_ldr_profile={name:test_data_profile};
        this.im_seqcls_testing_tasks[name].meta_ldr_profile={name:testing_meta_profile};
        deps =test_data_profile[neko_datastream_profile.ENDPOINTS];

        # don't make more than one data stream for testing unless you are up to some testing time per batch  referenced calibration.
        # but that will be far beyond the scope of 32x.
        # srsly i can think of a few use cases--- but hu cares except very niche ocr ppl?
        assert (len(deps)==1);
        dldr_ep=deps[0];
        data_mod_prfx=test_data_profile[neko_datastream_profile.SIGNATURES][dldr_ep];
        # note mod configs are trees--
        # basic sharing controls are done through global shared models
        # Advanced sharing controls are done through bogomods
        dpcfg= neko_datastream_processing_profile.get_default_word_process(dldr_ep,data_mod_prfx, this.net_size_meta.item_data_size);
        this.im_seqcls_testing_tasks[name].data_process_profile = dpcfg;

        meps = testing_meta_profile[neko_meta_stream_profile.ENDPOINTS];

        # well you may want to ensemble/choose from different metas, but that aint happen in 32x
        assert (len(meps)==1);
        mldr_ep =meps[0];
        msig=testing_meta_profile[neko_meta_stream_profile.SIDE_INFO_MBND][mldr_ep];
        mpcfg= neko_metastream_processing.get_default_im_meta_process(mldr_ep, 128,this.net_size_meta.item_data_size,msig);
        this.im_seqcls_testing_tasks[name].meta_process_profile = mpcfg;

        roi_eps= dpcfg[neko_datastream_processing_profile.ROI_EPS];
        assert (len(roi_eps)==1); # this config does not allow routing
        ap_eps=mpcfg[neko_metastream_processing.MODSCP_ENDPOINTS];
        assert (len(ap_eps)==1); # this config does not allow meta selection, routing, or ensembling

        this.im_seqcls_testing_tasks[name].task_mod_name =task_mod_prfx;
        this.im_seqcls_testing_tasks[name].task_process_profile[name]=neko_task_processing.get_default_seq_testing_task(
            roi_eps[0], ap_eps[0], name,task_mod_prfx,maxT=32);


        pass;
