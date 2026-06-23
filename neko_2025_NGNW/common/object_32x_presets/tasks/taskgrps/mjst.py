

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
def arm_mjst_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome):
    EN_META = "meta-endigit_uncased";
    EN_WORD_DATA = "mjst";
    EN_WORD_DATA_SIG = "en_synth_word";
    EN_TSK_ZSOS="en_oszsl";
    EN_SINFO_NOTO_SIG="noto_endigit"
    FPBPG="mjst";

    mpath_en = os.path.join(pcfg.data_root, "dicts_v3/endigit_uncased/");
    mprof_en= neko_meta_stream_profile.get_default_training_loader_noto(
            EN_META, mpath_en, EN_WORD_DATA, mpath_en,EN_SINFO_NOTO_SIG);
    tome.mount_im_meta_stream(EN_META,mprof_en,FPBPG);

    dprofile_mjst=neko_datastream_profile.get_default_training_im_seq_loader(
        EN_WORD_DATA, pcfg.data_root,get_abi_mjst(pcfg.data_root),EN_WORD_DATA_SIG, batchsize=64,maxT=32);
    tome.mount_word_datastream_im_seq(EN_WORD_DATA,dprofile_mjst,FPBPG);

    tome.add_simple_osseq_task(tome.seq_roi_eps[EN_WORD_DATA][0],tome.meta_eps[EN_META][0],EN_TSK_ZSOS,None,FPBPG,maxT=32);

    te_mprof_enchs = neko_meta_stream_profile.get_default_test_loader_noto(EN_META, mpath_en, mpath_en,
                                                                           EN_SINFO_NOTO_SIG);

    te_dprofile_iiit5k = neko_datastream_profile.get_default_testing_im_seq_loader(EN_WORD_DATA,
                                                                                   get_enword_te(pcfg.data_root),
                                                                                   EN_WORD_DATA_SIG, 32);
    # for testing, we only do it in a fixed batchsize--- bcs some alive protocols actually depend on that to work.
    # n
    tome.mount_testing_word_osr_im("iiit5k_val", te_dprofile_iiit5k, te_mprof_enchs, EN_TSK_ZSOS,True);
    return tome;

