
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.log import fatal

from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile

import os


from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome




def arm_egptology_v3R_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,aug=None,tie_weight=False):

    EGPT_META_VAL = "meta-ecr-v3R-val"; # full char in val, for fsl split
    EGPT_META_TR = "meta-ecr-v3R-tr"; # char in tra only, for OSR split
    EGPT_META_TEST="meta-ecr-v3R-test";
    
    EGPT_CHAR_DATA = "ecr-v3R";
    EGPT_CHAR_DATA_SIG="ecr-v3R_data"; # what model group shall we use.
    EGPT_SINFO_SIG = "ecr-v3R_sinfo";

    if(tie_weight):
        EGPT_SINFO_MOD_SIG = EGPT_CHAR_DATA_SIG;
    else:
        EGPT_SINFO_MOD_SIG = "ecr-v3R_sinfo";
    EGPT_TSK_FSOS="ecr-v3R_oszsl";
    FPBPG="ecr-v3R";

    trd= {
        "typename": "ecr_v3R_tr",
        "dict": {
            "ecr_v3R_tr": os.path.join(pcfg.data_root,"ecr-v3R/ecr_v3_train/"),
        }
    }
    if(aug=="abinet"):
        dprofile_egpch=neko_datastream_profile.get_abi_aug_training_im_item_loader(
            EGPT_CHAR_DATA, pcfg.data_root,trd,EGPT_CHAR_DATA_SIG, batchsize=128);
    elif aug is None:
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            EGPT_CHAR_DATA, pcfg.data_root, trd, EGPT_CHAR_DATA_SIG, batchsize=128);
    else:
        fatal("wut");
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            EGPT_CHAR_DATA, pcfg.data_root, trd, EGPT_CHAR_DATA_SIG, batchsize=128);

    tome.mount_datastream_im_item(EGPT_CHAR_DATA,dprofile_egpch,FPBPG);

    mpath_ecr_tr = os.path.join(pcfg.data_root, "dicts_v3/ecr-v3R/ecr_v3_train/");
    mpath_ecr_val = os.path.join(pcfg.data_root, "dicts_v3/ecr-v3R/ecr_v3_val/");
    mpath_ecr_te = os.path.join(pcfg.data_root, "dicts_v3/ecr-v3R/ecr_v3_test/");

    mprof_ecr=neko_meta_stream_profile.get_default_training_loader_sample_kshot(
        EGPT_META_TR, mpath_ecr_tr,
        EGPT_CHAR_DATA,
        os.path.join(pcfg.data_root, "ecr-v3R/ecr_v3_train/"),EGPT_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight
    );
    tome.mount_im_meta_stream(EGPT_META_TR,mprof_ecr,FPBPG);


    tome.add_simple_oscr_task(tome.item_roi_eps[EGPT_CHAR_DATA][0],tome.meta_eps[EGPT_META_TR][0],EGPT_TSK_FSOS,None,FPBPG);

    # OSR uses training character as support for seen and first occurance on testing set for novel---
    # but hey if the support sample is crappy it will be a shitshow ofc
    val_mprof_ecr_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_VAL, mpath_ecr_val,os.path.join(pcfg.data_root,"ecr-v3R/ecr_v3_val_support/"),
        EGPT_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight);

    te_mprof_ecr_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TEST, mpath_ecr_te,os.path.join(pcfg.data_root,"ecr-v3R/ecr_v3_test_support/"),
        EGPT_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight);
    # OSR uses training character as support--- but hey if the support sample is crappy it will be a shitshow ofc
    val_mprof_ecr_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TR + "val", mpath_ecr_tr, os.path.join(pcfg.data_root, "ecr-v3R/ecr_v3_tra_support/"),
        EGPT_SINFO_MOD_SIG, imshot_k=1,mod_tied=tie_weight);

    te_mprof_ecr_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TR+"te", mpath_ecr_tr, os.path.join(pcfg.data_root, "ecr-v3R/ecr_v3_tra_support/"),
        EGPT_SINFO_MOD_SIG, imshot_k=1,mod_tied=tie_weight);
    vald={
        "typename": "ecr_v3R_val",
        "dict": {
            "ecr_v3R_val":os.path.join(pcfg.data_root,"ecr-v3R/ecr_v3_val/"),
        }
    };
    val_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(EGPT_CHAR_DATA,
                                                                                   vald,
                                                                                   EGPT_CHAR_DATA_SIG, 32);

    ted={
        "typename": "ecr_v3R_te",
        "dict": {
            "ecr_v3R_te":os.path.join(pcfg.data_root,"ecr-v3R/ecr_v3_test/"),
        }
    };
    te_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(EGPT_CHAR_DATA,
                                                                                   ted,
                                                                                   EGPT_CHAR_DATA_SIG, 32);
    tome.mount_testing_char_osr_im("ecr-v3R_val_fsl_1shot", val_dprofile_ctwch, val_mprof_ecr_fsl, EGPT_TSK_FSOS,False);
    tome.mount_testing_char_osr_im("ecr-v3R_val_osr_1shot", val_dprofile_ctwch, val_mprof_ecr_osr, EGPT_TSK_FSOS,False);




    tome.mount_testing_char_osr_im("ecr-v3R_test_fsl_1shot", te_dprofile_ctwch, te_mprof_ecr_fsl, EGPT_TSK_FSOS,False);
    tome.mount_testing_char_osr_im("ecr-v3R_test_osr_1shot", te_dprofile_ctwch, te_mprof_ecr_osr, EGPT_TSK_FSOS,False);


    return tome;

