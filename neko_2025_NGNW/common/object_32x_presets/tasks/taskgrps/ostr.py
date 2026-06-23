

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.templates.lsct_stream_profiles import neko_lsct_ho_repeater_template, \
    neko_lsct_ho_shuff_template

from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome

import os

from neko_sdk.log import fatal


def get_test_jpn_hori(dataroot):
    return {
        "typename": "word_te",
        "dict": {
            "jpn_hori": os.path.join(dataroot,"mlttrjp_hori"),
        }
    };
def get_test_kr_hori(dataroot):
    return {
        "typename": "word_te",
        "dict": {
            "kr_hori": os.path.join(dataroot,"mlttrkr_hori"),
        }
    };
def get_test_synyi_hori(dataroot):
    return {
        "typename": "word_te",
        "dict": {
            "yi_synth": os.path.join(dataroot,"synthyi_5k"),
        }
    };


def arm_chs_jpnkr_hori_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=4,self_repeat=False,lsct=False,lsct_sem=False,lsct_ins=False,xcts_weight=-1,lsct_fsz=32,lsct_spacing=None,    FPBPG="ostrv1",capacity=128):

    OSTRV1_META_TR = "meta-ostrv1-tr"; # char in tra only, for OSR split

    OSTRV1_DATA = "ostrv1";
    OSTRV1_DATA_SIG="ostrv1_data"; # what model group shall we use.

    OSTRV1_SINFO_NOTO_SIG="ostrv1_sinfo";
    OSTRV1_XTCS_TASK_SIG= "ostrv1_cross_task_cosine_sim_task";
    OSTRV1_TASK_SIG= "ostrv1_task";

    OSTRV1_LSCT_TSK_ZSOS="ostrv1_lsct_task_zsos";
    OSTRV1_LSCT_TSK_SEMSEG="ostrv1_lsct_task_semseg";
    OSTRV1_LSCT_TSK_CA_INSSEG="ostrv1_lsct_task_ca_insseg";

    LSCT_OSTR="lsct_ostr";

    trd= {
        "typename": "ostrv1_v2_tr",
        "dict": {
            "ostrv1_tr_art": os.path.join(pcfg.data_root,"artdb_seen/"),
            "ostrv1_tr_ctw": os.path.join(pcfg.data_root, "ctwdb_seen/"),
            "ostrv1_tr_lsvt": os.path.join(pcfg.data_root, "lsvtdb_seen/"),
            "ostrv1_tr_mlttrchlat": os.path.join(pcfg.data_root, "mlttrchlat_seen/"),
            "ostrv1_tr_rctwtr": os.path.join(pcfg.data_root, "rctwtrdb_seen/"),
        }
    };

    mpath_enchs = os.path.join(pcfg.data_root, "dicts_v3/chs3755_en_uncased_digits/");
    mprof_enchs=neko_meta_stream_profile.get_default_training_loader_noto(
        OSTRV1_META_TR, mpath_enchs,
        OSTRV1_DATA,
        mpath_enchs,OSTRV1_SINFO_NOTO_SIG
    );
    tome.mount_im_meta_stream(OSTRV1_META_TR,mprof_enchs,FPBPG);
    xcts_items=['meta-ostrv1-tr-sampled-noto-armed'];
    xcts_seqs=["ostrv1-abiaug"];

    dprofile=neko_datastream_profile.get_default_training_im_seq_loader(
        OSTRV1_DATA, pcfg.data_root,trd,OSTRV1_DATA_SIG, batchsize=bsf*16);
    tome.mount_word_datastream_im_seq(OSTRV1_DATA,dprofile,FPBPG);
    tome.add_simple_osseq_task(
        tome.seq_roi_eps[OSTRV1_DATA][0],tome.meta_eps[OSTRV1_META_TR][0],
        OSTRV1_TASK_SIG,None,FPBPG,maxT=32);

    if(lsct):
        xcts_seqs.append(LSCT_OSTR);
        if(lsct==True or lsct=="lsct_repeater"):
            tome.mount_lsct_seq_stream(LSCT_OSTR,neko_lsct_ho_repeater_template.get_default_lsct_repeater(
                    OSTRV1_DATA, pcfg.data_root, LSCT_OSTR,lsct_fsz,lsct_spacing),FPBPG);

        elif (lsct=="stream_fork"):
            fatal("not impl");
        elif (lsct=="lsct_shuff"):
            tome.mount_lsct_seq_stream(LSCT_OSTR, neko_lsct_ho_shuff_template.get_default_lsct_shuff(
                OSTRV1_DATA, pcfg.data_root, LSCT_OSTR, lsct_fsz,lsct_spacing), FPBPG);
        else:
            fatal("unknown lsct type");
        tome.add_simple_osseq_task(tome.seq_roi_eps[LSCT_OSTR][0],tome.meta_eps[OSTRV1_META_TR][0],OSTRV1_LSCT_TSK_ZSOS,None,FPBPG,maxT=32);
        if(lsct_sem):
            tome.add_simple_ossemseg_task(tome.feat_eps[LSCT_OSTR][0],tome.meta_eps[OSTRV1_META_TR][0],OSTRV1_LSCT_TSK_SEMSEG,None,FPBPG);
        if(lsct_ins):
            tome.add_ordered_cainsseg_task(tome.feat_eps[LSCT_OSTR][0],tome.meta_eps[OSTRV1_META_TR][0],OSTRV1_LSCT_TSK_CA_INSSEG,None,FPBPG);

    te_dprofile_jpn = neko_datastream_profile.get_default_testing_im_seq_loader(OSTRV1_DATA,
                                                                                   get_test_jpn_hori(pcfg.data_root),
                                                                                   OSTRV1_DATA_SIG, 32);

    if (xcts_weight>0):
        tome.add_simple_xtcs(xcts_seqs,xcts_items,[],os.path.join(mpath_enchs,"tdict.pt"),OSTRV1_XTCS_TASK_SIG,FPBPG,xcts_weight);
    OSTRV1_META_TEST_JPN_GZSL="meta-ostrv1-test_jpn_gzsl";
    mpath_jpn_gzsl = os.path.join(pcfg.data_root, "dicts_v3/dabjpmlt/");
    te_mprof_jpn_gzsl= neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_JPN_GZSL, mpath_jpn_gzsl, mpath_jpn_gzsl, OSTRV1_SINFO_NOTO_SIG);

    tome.mount_testing_word_osr_im("jpn_hori_gzsl", te_dprofile_jpn, te_mprof_jpn_gzsl, OSTRV1_TASK_SIG,True);


    # OSR (Open Set Recognition)
    OSTRV1_META_TEST_JPN_OSR = "meta-ostrv1-test_jpn_osr"
    mpath_jpn_osr = os.path.join(pcfg.data_root, "dicts_v3/dabjpmltch_osr/")
    te_mprof_jpn_osr = neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_JPN_OSR, mpath_jpn_osr, mpath_jpn_osr, OSTRV1_SINFO_NOTO_SIG);
    tome.mount_testing_word_osr_im("jpn_hori_osr", te_dprofile_jpn, te_mprof_jpn_osr, OSTRV1_TASK_SIG,
                                       False);


    # GOSR (Generalized Open Set Recognition)
    OSTRV1_META_TEST_JPN_GOSR = "meta-ostrv1-test_jpn_gosr"
    mpath_jpn_gosr = os.path.join(pcfg.data_root, "dicts_v3/dabjpmltch_nohirakata/")
    te_mprof_jpn_gosr = neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_JPN_GOSR, mpath_jpn_gosr, mpath_jpn_gosr, OSTRV1_SINFO_NOTO_SIG);
    tome.mount_testing_word_osr_im("jpn_hori_gosr", te_dprofile_jpn, te_mprof_jpn_gosr, OSTRV1_TASK_SIG,
                                       False);

    # OSTR (Open-Set Text Recognition)
    OSTRV1_META_TEST_JPN_OSTR = "meta-ostrv1-test_jpn_ostr"
    mpath_jpn_ostr = os.path.join(pcfg.data_root, "dicts_v3/dabjpmltch_kanji/")
    te_mprof_jpn_ostr = neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_JPN_OSTR, mpath_jpn_ostr, mpath_jpn_ostr, OSTRV1_SINFO_NOTO_SIG)
    tome.mount_testing_word_osr_im("jpn_hori_ostr", te_dprofile_jpn, te_mprof_jpn_ostr, OSTRV1_TASK_SIG,
                                       False);


    # --- Korean (KR) Datasets ---
    te_dprofile_kr = neko_datastream_profile.get_default_testing_im_seq_loader(OSTRV1_DATA,
                                                                               get_test_kr_hori(pcfg.data_root),
                                                                               OSTRV1_DATA_SIG, 32);

    # GZSL (Generalized Zero-Shot Learning)
    OSTRV1_META_TEST_KR_GZSL = "meta-ostrv1-test_kr_gzsl"
    mpath_kr_gzsl = os.path.join(pcfg.data_root, "dicts_v3/dabkrmlt/")
    te_mprof_kr_gzsl = neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_KR_GZSL, mpath_kr_gzsl, mpath_kr_gzsl, OSTRV1_SINFO_NOTO_SIG);

    tome.mount_testing_word_osr_im("kr_hori_gzsl", te_dprofile_kr, te_mprof_kr_gzsl, OSTRV1_TASK_SIG,
                                       True);

    # --- Yi (Yi) Datasets ---
    te_dprofile_yi_synth = neko_datastream_profile.get_default_testing_im_seq_loader(OSTRV1_DATA,
                                                                               get_test_synyi_hori(pcfg.data_root),
                                                                               OSTRV1_DATA_SIG, 32);

    # GZSL (Generalized Zero-Shot Learning)
    OSTRV1_META_TEST_YI_SYN_GZSL = "meta-ostrv1-test_synyi_gzsl"
    mpath_yi_gzsl = os.path.join(pcfg.data_root, "dicts_v3/yiall/")
    te_mprof_yi_syn_gzsl = neko_meta_stream_profile.get_default_test_loader_noto(
        OSTRV1_META_TEST_YI_SYN_GZSL, mpath_yi_gzsl, mpath_yi_gzsl, OSTRV1_SINFO_NOTO_SIG);

    tome.mount_testing_word_osr_im("syn_yi_hori_gzsl", te_dprofile_yi_synth, te_mprof_yi_syn_gzsl, OSTRV1_TASK_SIG,
                                   True);

    return tome;
