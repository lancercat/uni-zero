

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

def arm_ctwch_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,lsct=True,bsf=8):
    CHS_META = "meta-chs_3755";
    CH_CHAR_DATA = "ctwch";
    CH_CHAR_DATA_SIG="chs_real_char"; # what model group shall we use.


    CHS_SINFO_NOTO_SIG="3755_noto";
    CHS_SINFO_CTWIM_SIG="ctw_trim";
    CHS_TSK_ZSOS="ctw_oszsl";
    CHS_TSK_FSOS = "ctw_osfsl";
    CHS_LSCT_TSK_ZSOS="ctw_lsct_oszsl";
    CHS_LSCT_TSK_FSOS = "ctw_lsct_osfsl";

    LSCT_CH = P(CH_CHAR_DATA, "LSCT_repeat");
    LSCT_CH_DATA_SIG = P(CH_CHAR_DATA, "LSCT_repeat"); # lsct is di

    FPBPG="ctw_ch";


    dprofile_ctwch=neko_datastream_profile.get_default_training_im_item_loader(
        CH_CHAR_DATA, pcfg.data_root,get_ctw_yolox_tr(pcfg.data_root),CH_CHAR_DATA_SIG, batchsize=16*bsf);
    tome.mount_datastream_im_item(CH_CHAR_DATA,dprofile_ctwch,FPBPG);

    mpath_enchs = os.path.join(pcfg.data_root, "dicts_v3/chs3755_en_uncased_digits/");
    mprof_enchs=neko_meta_stream_profile.get_default_training_loader_noto_sample_kshot(
        CHS_META, mpath_enchs,
        CH_CHAR_DATA,
        mpath_enchs,CHS_SINFO_NOTO_SIG,
        os.path.join(pcfg.data_root, "ctw_ch_yolox/train/"),CHS_SINFO_CTWIM_SIG
    );
    tome.mount_im_meta_stream(CHS_META,mprof_enchs,FPBPG);

    tome.add_simple_oscr_task(tome.item_roi_eps[CH_CHAR_DATA][0],tome.meta_eps[CHS_META][0],CHS_TSK_ZSOS,None,FPBPG);
    tome.add_simple_oscr_task(tome.item_roi_eps[CH_CHAR_DATA][0],tome.meta_eps[CHS_META][1],CHS_TSK_FSOS,None,FPBPG);

    if(lsct):
        tome.mount_lsct_char_stream(LSCT_CH,neko_lsct_repeater_template.get_default_lsct_repeater(
                CH_CHAR_DATA, pcfg.data_root, LSCT_CH),FPBPG);

        tome.add_simple_oscr_task(tome.item_roi_eps[LSCT_CH][0],tome.meta_eps[CHS_META][0],CHS_LSCT_TSK_ZSOS,None,FPBPG);
        tome.add_simple_oscr_task(tome.item_roi_eps[LSCT_CH][0],tome.meta_eps[CHS_META][1],CHS_LSCT_TSK_FSOS,None,FPBPG);

    te_mprof_enchs_osl = neko_meta_stream_profile.get_default_testing_loader_sample_kshot(
        CHS_META, mpath_enchs,os.path.join(pcfg.data_root,"ctw_ch_yolox/train/"),   CHS_SINFO_CTWIM_SIG,imshot_k=1);
    te_dprofile_ctwch_osl = neko_datastream_profile.get_default_testing_im_item_loader(CH_CHAR_DATA,
                                                                                   get_ctw_yolox_val(pcfg.data_root),
                                                                                   CH_CHAR_DATA_SIG, 32);
    tome.mount_testing_char_osr_im("ctw_val_fsl_1shot", te_dprofile_ctwch_osl, te_mprof_enchs_osl, CHS_TSK_FSOS,False);

    # for testing, we only do it in a fixed batchsize--- bcs some alive protocols actually depend on that to work.

    te_mprof_enchs = neko_meta_stream_profile.get_default_test_loader_noto(CHS_META, mpath_enchs, mpath_enchs,
                                                                           CHS_SINFO_NOTO_SIG);
    te_dprofile_ctwch = neko_datastream_profile.get_default_testing_im_item_loader(CH_CHAR_DATA,
                                                                                   get_ctw_yolox_val(pcfg.data_root),
                                                                                   CH_CHAR_DATA_SIG, 32);

    tome.mount_testing_char_osr_im("ctw_val", te_dprofile_ctwch, te_mprof_enchs, CHS_TSK_ZSOS,False);

    return tome;

