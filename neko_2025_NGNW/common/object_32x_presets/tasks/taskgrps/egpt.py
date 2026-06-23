

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_sdk.neko_framework_NG.names import P
from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.lsct_stream_profiles import neko_lsct_repeater_template

import os
from osocrNG.data_utils.common_data_presets_mk5.data.presets.protocol.train_data_dict_collection import \
    get_ctw_yolox_tr,get_ctw_yolox_val,get_ctw_yolox_te,get_enword_te
from osocrNG.data_utils.common_data_presets_mk5.data.presets.protocol.train_data_dict_collection import get_abi_mjst

from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome
def arm_egptology_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome):

    EGPT_META_VAL = "meta-ecr-val"; # full char in val, for fsl split
    EGPT_META_TR = "meta-ecr-tr"; # char in tra only, for OSR split
    EGPT_META_TEST="meta-ecr-test";
    
    EGPT_CHAR_DATA = "ecr";
    EGPT_CHAR_DATA_SIG="ecr_data"; # what model group shall we use.

    EGPT_SINFO_SIG="ecr_sinfo";
    EGPT_TSK_FSOS="ecr_oszsl";
    FPBPG="ecr";

    trd= {
        "typename": "ecr_v2_tr",
        "dict": {
            "ecr_v2_tr": os.path.join(pcfg.data_root,"ecr/ecr_v2_train/"),
        }
    }
    dprofile_ctwch=neko_datastream_profile.get_default_training_im_item_loader(
        EGPT_CHAR_DATA, pcfg.data_root,trd,EGPT_CHAR_DATA_SIG, batchsize=128);
    tome.mount_datastream_im_item(EGPT_CHAR_DATA,dprofile_ctwch,FPBPG);

    mpath_ecr_tr = os.path.join(pcfg.data_root, "dicts_v3/ecr/ecr_v2_train/");
    mpath_ecr_val = os.path.join(pcfg.data_root, "dicts_v3/ecr/ecr_v2_val/");
    mpath_ecr_te = os.path.join(pcfg.data_root, "dicts_v3/ecr/ecr_v2_test/");

    mprof_enchs=neko_meta_stream_profile.get_default_training_loader_sample_kshot(
        EGPT_META_TR, mpath_ecr_tr,
        EGPT_CHAR_DATA,
        os.path.join(pcfg.data_root, "ecr/ecr_v2_train/"),EGPT_SINFO_SIG,imshot_k=1
    );
    tome.mount_im_meta_stream(EGPT_META_TR,mprof_enchs,FPBPG);


    tome.add_simple_oscr_task(tome.item_roi_eps[EGPT_CHAR_DATA][0],tome.meta_eps[EGPT_META_TR][0],EGPT_TSK_FSOS,None,FPBPG);

    val_mprof_enchs_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_VAL, mpath_ecr_val,os.path.join(pcfg.data_root,"ecr/ecr_v2_val_support/"),
        EGPT_SINFO_SIG,imshot_k=1);
    te_mprof_enchs_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TR+"te", mpath_ecr_tr, os.path.join(pcfg.data_root, "ecr/ecr_v2_tra_support/"),
        EGPT_SINFO_SIG, imshot_k=1);
    val_mprof_enchs_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TR+"val", mpath_ecr_tr, os.path.join(pcfg.data_root, "ecr/ecr_v2_tra_support/"),
        EGPT_SINFO_SIG, imshot_k=1);
    vald={
        "typename": "ecr_v2_val",
        "dict": {
            "ecr_v2_val":os.path.join(pcfg.data_root,"ecr/ecr_v2_val/"),
        }
    };
    val_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(EGPT_CHAR_DATA,
                                                                                   vald,
                                                                                   EGPT_CHAR_DATA_SIG, 32);

    ted={
        "typename": "ecr_v2_te",
        "dict": {
            "ecr_v2_te":os.path.join(pcfg.data_root,"ecr/ecr_v2_test/"),
        }
    };
    te_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(EGPT_CHAR_DATA,
                                                                                   ted,
                                                                                   EGPT_CHAR_DATA_SIG, 32);
    tome.mount_testing_char_osr_im("ecr_val_fsl_1shot", val_dprofile_ctwch, val_mprof_enchs_fsl, EGPT_TSK_FSOS,False);
    tome.mount_testing_char_osr_im("ecr_val_osr_1shot", val_dprofile_ctwch, val_mprof_enchs_osr, EGPT_TSK_FSOS,False);


    te_mprof_enchs_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        EGPT_META_TEST, mpath_ecr_te,os.path.join(pcfg.data_root,"ecr/ecr_v2_test_support/"),
        EGPT_SINFO_SIG,imshot_k=1);

    tome.mount_testing_char_osr_im("ecr_test_fsl_1shot", te_dprofile_ctwch, te_mprof_enchs_fsl, EGPT_TSK_FSOS,False);
    tome.mount_testing_char_osr_im("ecr_test_osr_1shot", te_dprofile_ctwch, te_mprof_enchs_osr, EGPT_TSK_FSOS,False);


    return tome;
