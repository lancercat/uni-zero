

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.log import fatal

from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile

import os


from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome



def arm_hustobc_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,aug=None,tie_weight=False,grey=False):

    HUSTOBC_META_VAL = "meta-hustobc-val"; # full char in val, for fsl split
    HUSTOBC_META_TR = "meta-hustobc-tr"; # char in tra only, for OSR split
    HUSTOBC_META_TEST="meta-hustobc-test";
    
    HUSTOBC_CHAR_DATA = "hustobc";
    HUSTOBC_CHAR_DATA_SIG="hustobc_data"; # what model group shall we use.
    HUSTOBC_SINFO_SIG = "hustobc_sinfo";

    if(tie_weight):
        HUSTOBC_SINFO_MOD_SIG = HUSTOBC_CHAR_DATA_SIG;
    else:
        HUSTOBC_SINFO_MOD_SIG = "hustobc_sinfo";
    HUSTOBC_TSK_FSOS="hustobc_oszsl";
    FPBPG="hustobc";

    trd= {
        "typename": "hustobc_tr",
        "dict": {
            "hustobc_tr": os.path.join(pcfg.data_root,"hustobc/hust-obctrain/"),
        }
    }
    if(aug=="abinet"):
        dprofile_egpch=neko_datastream_profile.get_abi_aug_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root,trd,HUSTOBC_CHAR_DATA_SIG, batchsize=128,force_grey_scale=grey);
    elif aug is None:
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root, trd, HUSTOBC_CHAR_DATA_SIG, batchsize=128,force_grey_scale=grey);
    else:
        fatal("wut");
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root, trd, HUSTOBC_CHAR_DATA_SIG, batchsize=128,force_grey_scale=grey);

    tome.mount_datastream_im_item(HUSTOBC_CHAR_DATA,dprofile_egpch,FPBPG);

    mpath_hustobc_tr = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obctrain/");
    mpath_hustobc_val = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obcval/");
    mpath_hustobc_te = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obctest/");

    mprof_ecr=neko_meta_stream_profile.get_default_training_loader_sample_kshot(
        HUSTOBC_META_TR, mpath_hustobc_tr,
        HUSTOBC_CHAR_DATA,
        os.path.join(pcfg.data_root, "hustobc/hust-obctrain/"),HUSTOBC_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight,force_grey=grey
    );
    tome.mount_im_meta_stream(HUSTOBC_META_TR,mprof_ecr,FPBPG);


    tome.add_simple_oscr_task(tome.item_roi_eps[HUSTOBC_CHAR_DATA][0],tome.meta_eps[HUSTOBC_META_TR][0],HUSTOBC_TSK_FSOS,None,FPBPG);

    # OSR uses training character as support for seen and first occurance on testing set for novel---
    # but hey if the support sample is crappy it will be a shitshow ofc
    val_mprof_hustobc_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        HUSTOBC_META_VAL, mpath_hustobc_val,os.path.join(pcfg.data_root,"hustobc/hust-obcval_support/"),
        HUSTOBC_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight,force_grey=grey);

    te_mprof_hustobc_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        HUSTOBC_META_TEST, mpath_hustobc_te,os.path.join(pcfg.data_root,"hustobc/hust-obctest_support/"),
        HUSTOBC_SINFO_MOD_SIG,imshot_k=1,mod_tied=tie_weight,force_grey=grey);

    vald={
        "typename": "hustobc_val",
        "dict": {
            "hustobc_val":os.path.join(pcfg.data_root,"hustobc/hust-obcval/"),

        }
    };

    val_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                   vald,
                                                                                   HUSTOBC_CHAR_DATA_SIG, 32,force_grey_scale=grey);

    ted={
        "typename": "hustobc_te",
        "dict": {
            "hustobc_te":os.path.join(pcfg.data_root,"hustobc/hust-obctest/"),
        }
    };
    te_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                   ted,
                                                                                   HUSTOBC_CHAR_DATA_SIG, 32,force_grey_scale=grey);
    tome.mount_testing_char_osr_im("hustobc_val_fsl_1shot", val_dprofile_ctwch, val_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,False);




    tome.mount_testing_char_osr_im("hustobc_test_fsl_1shot", te_dprofile_ctwch, te_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,False);

    #
    # # OSR uses training character as support--- but hey if the support sample is crappy it will be a shitshow ofc
    # val_mprof_hustobc_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
    #     HUSTOBC_META_TR + "val", mpath_hustobc_tr, os.path.join(pcfg.data_root, "hustobc/hust-obctrain_support/"),
    #     HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);
    #
    # te_mprof_hustobc_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
    #     HUSTOBC_META_TR + "te", mpath_hustobc_tr, os.path.join(pcfg.data_root, "hustobc/hust-obctrain_support/"),
    #     HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);
    # tome.mount_testing_char_osr_im("hustobc_val_osr_1shot", val_dprofile_ctwch, val_mprof_hustobc_osr, HUSTOBC_TSK_FSOS,False);
    #
    #
    # tome.mount_testing_char_osr_im("hustobc_test_osr_1shot", te_dprofile_ctwch, te_mprof_hustobc_osr, HUSTOBC_TSK_FSOS,False);
    #

    return tome;


def arm_hustobc_zero_task_grp(pcfg: neko_platform_cfg, tome: neko_320_tome, aug=None, tie_weight=False, grey=False):
    HUSTOBC_META_VAL = "meta-hustobc-val";  # full char in val, for fsl split
    HUSTOBC_META_TR = "meta-hustobc-tr";  # char in tra only, for OSR split
    HUSTOBC_META_TEST = "meta-hustobc-test";

    HUSTOBC_CHAR_DATA = "hustobc_zero_0_299";
    HUSTOBC_CHAR_DATA_SIG = "hustobc_data";  # what model group shall we use.
    HUSTOBC_SINFO_SIG = "hustobc_sinfo";

    if (tie_weight):
        HUSTOBC_SINFO_MOD_SIG = HUSTOBC_CHAR_DATA_SIG;
    else:
        HUSTOBC_SINFO_MOD_SIG = "hustobc_sinfo";
    HUSTOBC_TSK_FSOS = "hustobc_oszsl";
    FPBPG = "hustobc";

    trd = {
        "typename": "hustobc_tr_zero_0_299",
        "dict": {
            "hustobc_tr_zero_0_299": os.path.join(pcfg.data_root, "hustobc/hust-obc-zerotrain-igen-0-299/"),
        }
    }
    if (aug == "abinet"):
        dprofile_egpch = neko_datastream_profile.get_abi_aug_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root, trd, HUSTOBC_CHAR_DATA_SIG, batchsize=128, force_grey_scale=grey);
    elif aug is None:
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root, trd, HUSTOBC_CHAR_DATA_SIG, batchsize=128, force_grey_scale=grey);
    else:
        fatal("wut");
        dprofile_egpch = neko_datastream_profile.get_default_training_im_item_loader(
            HUSTOBC_CHAR_DATA, pcfg.data_root, trd, HUSTOBC_CHAR_DATA_SIG, batchsize=128, force_grey_scale=grey);

    tome.mount_datastream_im_item(HUSTOBC_CHAR_DATA, dprofile_egpch, FPBPG);

    mpath_hustobc_tr = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obc-zerotrain-igen-0-299/");
    mpath_hustobc_val = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obcval/"); # we use training support samples
    mpath_hustobc_te = os.path.join(pcfg.data_root, "dicts_v3/hustobc/hust-obctest/");

    mprof_ecr = neko_meta_stream_profile.get_default_training_loader_sample_kshot(
        HUSTOBC_META_TR, mpath_hustobc_tr,
        HUSTOBC_CHAR_DATA,
        os.path.join(pcfg.data_root, "hustobc/hust-obctrain/"), HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight,
        force_grey=grey
    );
    tome.mount_im_meta_stream(HUSTOBC_META_TR, mprof_ecr, FPBPG);

    tome.add_simple_oscr_task(tome.item_roi_eps[HUSTOBC_CHAR_DATA][0], tome.meta_eps[HUSTOBC_META_TR][0],
                              HUSTOBC_TSK_FSOS, None, FPBPG);

    # OSR uses training character as support for seen and first occurance on testing set for novel---
    # but hey if the support sample is crappy it will be a shitshow ofc
    val_mprof_hustobc_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        HUSTOBC_META_VAL, mpath_hustobc_val, os.path.join(pcfg.data_root, "hustobc/hust-obcval_support/"),
        HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);

    te_mprof_hustobc_fsl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        HUSTOBC_META_TEST, mpath_hustobc_te, os.path.join(pcfg.data_root, "hustobc/hust-obctest_support/"),
        HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);

    vald_gzsl = {
        "typename": "hustobc_val_gzsl",
        "dict": {
            "hustobc_val_gzsl": os.path.join(pcfg.data_root, "hustobc/hust-obcval/"),
    }
    };
    vald_zero = {
        "typename": "hustobc_val_zero",
        "dict": {
            "hustobc_val_zero":os.path.join(pcfg.data_root, "hustobc/hust-obc-zeroval-0-299/"),
    }
    };
    vald_seen = {
        "typename": "hustobc_val",
        "dict": {
            "hustobc_val_seen": os.path.join(pcfg.data_root, "hustobc/hust-obc-zeroval-igen-0-299/")
    }
    };
    val_dprofile_gzsl = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    vald_gzsl,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    val_dprofile_zero = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    vald_zero,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    val_dprofile_seen = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    vald_seen,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    tome.mount_testing_char_osr_im("hustobc_val_fsl_gssl", val_dprofile_gzsl, val_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);
    tome.mount_testing_char_osr_im("hustobc_val_fsl_gssl_unseen", val_dprofile_zero, val_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);
    tome.mount_testing_char_osr_im("hustobc_val_fsl_gssl_seen", val_dprofile_seen, val_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);



    testd_gzsl = {
        "typename": "hustobc_test_gzsl",
        "dict": {
            "hustobc_test_gzsl": os.path.join(pcfg.data_root, "hustobc/hust-obctest/"),
    }
    };
    testd_zero = {
        "typename": "hustobc_test_zero",
        "dict": {
            "hustobc_test_zero":os.path.join(pcfg.data_root, "hustobc/hust-obc-zerotest-0-299/"),
    }
    };
    testd_seen = {
        "typename": "hustobc_test",
        "dict": {
            "hustobc_test_seen": os.path.join(pcfg.data_root, "hustobc/hust-obc-zerotest-igen-0-299/")
    }
    };
    test_dprofile_gzsl = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    testd_gzsl,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    test_dprofile_zero = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    testd_zero,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    test_dprofile_seen = neko_datastream_profile.get_default_testing_im_item_loader(HUSTOBC_CHAR_DATA,
                                                                                    testd_seen,
                                                                                    HUSTOBC_CHAR_DATA_SIG, 32,
                                                                                    force_grey_scale=grey);
    tome.mount_testing_char_osr_im("hustobc_test_fsl_gssl", test_dprofile_gzsl, te_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);
    tome.mount_testing_char_osr_im("hustobc_test_fsl_gssl_unseen", test_dprofile_zero, te_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);
    tome.mount_testing_char_osr_im("hustobc_test_fsl_gssl_seen", test_dprofile_seen, te_mprof_hustobc_fsl, HUSTOBC_TSK_FSOS,
                                   False);

    #
    # # OSR uses training character as support--- but hey if the support sample is crappy it will be a shitshow ofc
    # val_mprof_hustobc_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
    #     HUSTOBC_META_TR + "val", mpath_hustobc_tr, os.path.join(pcfg.data_root, "hustobc/hust-obctrain_support/"),
    #     HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);
    #
    # te_mprof_hustobc_osr = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
    #     HUSTOBC_META_TR + "te", mpath_hustobc_tr, os.path.join(pcfg.data_root, "hustobc/hust-obctrain_support/"),
    #     HUSTOBC_SINFO_MOD_SIG, imshot_k=1, mod_tied=tie_weight, force_grey=grey);
    # tome.mount_testing_char_osr_im("hustobc_val_osr_1shot", val_dprofile_ctwch, val_mprof_hustobc_osr, HUSTOBC_TSK_FSOS,False);
    #
    #
    # tome.mount_testing_char_osr_im("hustobc_test_osr_1shot", te_dprofile_ctwch, te_mprof_hustobc_osr, HUSTOBC_TSK_FSOS,False);
    #

    return tome;

