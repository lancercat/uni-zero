from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile

import os


from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome
from neko_2025_NGNW.common.object_32x_presets.templates.lsct_stream_profiles import neko_lsct_ho_repeater_template,    neko_lsct_ho_shuff_template

def get_ist_lang(lang,split,root):
    return{
        "typename": "istr_"+lang+"_tr",
        "dict": {
            "istr_"+lang+"_tr":os.path.join(root,"istr",lang+"_"+split),
        }
    }

def arm_ist_test_task(pcfg:neko_platform_cfg,tome:neko_320_tome,lang,trtag):
    ISTR_LANG_META_TE = "meta-" + lang+"-test";
    ISTR_LANG_META_VAL = "meta-" + lang+"-val"; # in theory you can re use, but you need to hack alot of cfgs, so don't

    ISTR_LANG_WORD_DATA = lang;
    ISTR_LANG_WORD_DATA_SIG = "istr_" + trtag;
    ISTR_LANG_TSK_ZSOS = trtag + "_oszsl";
    ISTR_LANG_SINFO_NOTO_SIG = "noto_" + trtag;
    mpath_istr = os.path.join(pcfg.data_root, "dicts_v3/istr/dab_istr_" + lang);

    te_mprof = neko_meta_stream_profile.get_default_test_loader_noto(ISTR_LANG_META_TE, mpath_istr, mpath_istr,
                                                                        ISTR_LANG_SINFO_NOTO_SIG);

    val_dprofile = neko_datastream_profile.get_default_testing_im_seq_loader(ISTR_LANG_WORD_DATA,
                                                                             get_ist_lang(lang, "val", pcfg.data_root),
                                                                             ISTR_LANG_WORD_DATA_SIG, 32);


    val_mprof = neko_meta_stream_profile.get_default_test_loader_noto(ISTR_LANG_META_VAL, mpath_istr, mpath_istr,
                                                                        ISTR_LANG_SINFO_NOTO_SIG);

    te_dprofile = neko_datastream_profile.get_default_testing_im_seq_loader(ISTR_LANG_WORD_DATA,
                                                                            get_ist_lang(lang, "test", pcfg.data_root),
                                                                            ISTR_LANG_WORD_DATA_SIG, 32);
    tome.mount_testing_word_osr_im("istr" + "val_" + lang, val_dprofile, te_mprof, ISTR_LANG_TSK_ZSOS, False);
    tome.mount_testing_word_osr_im("istr" + "test_" + lang, te_dprofile, val_mprof, ISTR_LANG_TSK_ZSOS, False);
    return tome;


def arm_ist_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,lang,train=True,testval=True,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None):
    ISTR_LANG_META = "meta-"+lang;
    ISTR_LANG_WORD_DATA = lang;

    ISTR_LANG_WORD_DATA_SIG = "istr_"+lang;
    ISTR_LANG_TSK_ZSOS=lang+"_oszsl";
    ISTR_LANG_SINFO_NOTO_SIG="noto_"+lang;
    ISTR_LSCT_LANG_TSK_ZSOS=ISTR_LANG_TSK_ZSOS+"_lsct";
    ISTR_LANG_WORD_DATA_LSCT = lang+"_lsct";

    FPBPG="istr"+lang;
    if (lsct_spacing is None):
        lsct_spacing=[0];
    mpath_istr = os.path.join(pcfg.data_root, "dicts_v3/istr/dab_istr_" + lang);
    if(train):
        mprof_istr = neko_meta_stream_profile.get_default_training_loader_noto(
            ISTR_LANG_META, mpath_istr, ISTR_LANG_WORD_DATA, mpath_istr, ISTR_LANG_SINFO_NOTO_SIG);
        tome.mount_im_meta_stream(ISTR_LANG_META, mprof_istr, FPBPG);

        dprofile_mjst=neko_datastream_profile.get_default_training_im_seq_loader(
            ISTR_LANG_WORD_DATA, pcfg.data_root,get_ist_lang(lang,"train",pcfg.data_root),ISTR_LANG_WORD_DATA_SIG, batchsize=bsf*8,maxT=32);
        tome.mount_word_datastream_im_seq(ISTR_LANG_WORD_DATA,dprofile_mjst,FPBPG);

        tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA][0],tome.meta_eps[ISTR_LANG_META][0],ISTR_LANG_TSK_ZSOS,None,FPBPG,maxT=32);
        if (lsct):
            if (lsct == True or lsct == "lsct_repeater"):
                tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT, neko_lsct_ho_repeater_template.get_default_lsct_repeater(
                    ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz, lsct_spacing), FPBPG);
            elif (lsct == "lsct_shuff"):
                tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT, neko_lsct_ho_shuff_template.get_default_lsct_shuff(
                    ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz, lsct_spacing), FPBPG);
            else:
                fatal("unknown lsct type");
            tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                       ISTR_LSCT_LANG_TSK_ZSOS, None, FPBPG, maxT=32);
            # if (lsct_sem):
            #     tome.add_simple_ossemseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
            #                                   OSTRV1_LSCT_TSK_SEMSEG, None, FPBPG);
            # if (lsct_ins):
            #     tome.add_ordered_cainsseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[OSTRV1_META_TR][0],
            #                                    OSTRV1_LSCT_TSK_CA_INSSEG, None, FPBPG)

    if(testval):
        val_mprof = neko_meta_stream_profile.get_default_test_loader_noto(ISTR_LANG_META+"-val", mpath_istr, mpath_istr,
                                                                               ISTR_LANG_SINFO_NOTO_SIG);
        te_mprof = neko_meta_stream_profile.get_default_test_loader_noto(ISTR_LANG_META+"-test", mpath_istr, mpath_istr,
                                                                               ISTR_LANG_SINFO_NOTO_SIG);
        val_dprofile = neko_datastream_profile.get_default_testing_im_seq_loader(ISTR_LANG_WORD_DATA,
                                                                                       get_ist_lang(lang,"val",pcfg.data_root),
                                                                                       ISTR_LANG_WORD_DATA_SIG, 32);

        te_dprofile = neko_datastream_profile.get_default_testing_im_seq_loader(ISTR_LANG_WORD_DATA,
                                                                                       get_ist_lang(lang,"test",pcfg.data_root),
                                                                                       ISTR_LANG_WORD_DATA_SIG, 32);
        tome.mount_testing_word_osr_im("istr"+"val_"+lang, val_dprofile, te_mprof, ISTR_LANG_TSK_ZSOS,False);
        tome.mount_testing_word_osr_im("istr"+"test_"+lang, te_dprofile, val_mprof, ISTR_LANG_TSK_ZSOS,False);
    return tome;

LANGS=[
    "bengali",
    "gujarati",
    "hindi",
    "kannada",
    "malayalam",
    "marathi",
    "punjabi",
    "tamil",
    "telugu",
    "odia"
];


def get_istr_zsl_bengali_gujarati(split,root):
    d={
        "typename": "get_istr_zsl_bengali_gujarati",
        "dict": {
        }
    };
    for lang in [
        "hindi",
        "kannada",
        "malayalam",
        "marathi",
        "punjabi",
        "tamil",
        "telugu",
        "odia"]:
        d["dict"]["istr_" + lang + "_tr"]=os.path.join(root, "istr", lang + "_" + split);
    return d;
def get_istr_zsl_bengali_gujarati_no_pt(split,root):
    d={
        "typename": "get_istr_zsl_bengali_gujarati",
        "dict": {
        }
    };
    for lang in [
        "hindi",
        "kannada",
        "malayalam",
        "marathi",
        "telugu",
        "odia"]:
        d["dict"]["istr_" + lang + "_tr"]=os.path.join(root, "istr", lang + "_" + split);
    return d;

def arm_istr_zslc_bengali_gujarati_tr(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2):
    lang="notbg"
    ISTR_LANG_META = "meta-"+lang;
    ISTR_LANG_WORD_DATA = lang;
    ISTR_LANG_WORD_DATA_SIG = "istr_"+lang;
    ISTR_LANG_TSK_ZSOS=lang+"_oszsl";
    ISTR_LANG_SINFO_NOTO_SIG="noto_"+lang;
    FPBPG="istr"+lang;
    mpath_istr = os.path.join(pcfg.data_root, "dicts_v3/istr/dab_istr_" + lang);
    mprof_istr = neko_meta_stream_profile.get_default_training_loader_noto(
        ISTR_LANG_META, mpath_istr, ISTR_LANG_WORD_DATA, mpath_istr, ISTR_LANG_SINFO_NOTO_SIG);
    tome.mount_im_meta_stream(ISTR_LANG_META, mprof_istr, FPBPG);

    dprofile_mjst=neko_datastream_profile.get_default_training_im_seq_loader(
        ISTR_LANG_WORD_DATA, pcfg.data_root,get_istr_zsl_bengali_gujarati("train",pcfg.data_root),ISTR_LANG_WORD_DATA_SIG, batchsize=bsf*16,maxT=32);
    tome.mount_word_datastream_im_seq(ISTR_LANG_WORD_DATA,dprofile_mjst,FPBPG);

    tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA][0],tome.meta_eps[ISTR_LANG_META][0],ISTR_LANG_TSK_ZSOS,None,FPBPG,maxT=32);

    return tome;
def arm_istr_zslc_bengali_gujarati_tr_2(pcfg: neko_platform_cfg, tome: neko_320_tome, bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False,capacity=128):
    lang = "notbg"
    ISTR_LANG_META = "meta-" + lang;
    ISTR_LANG_WORD_DATA = lang;
    ISTR_LANG_WORD_DATA_SIG = "istr_" + lang;
    ISTR_LANG_TSK_ZSOS = lang + "_oszsl";
    ISTR_LANG_SINFO_NOTO_SIG = "noto_" + lang;
    FPBPG = "istr" + lang;
    ISTR_LSCT_LANG_TSK_ZSOS=ISTR_LANG_TSK_ZSOS+"_lsct";
    ISTR_LSCT_LANG_TSK_SEMSEG=ISTR_LSCT_LANG_TSK_ZSOS+"_semseg";
    ISTR_LSCT_LANG_TSK_INSSEG=ISTR_LSCT_LANG_TSK_ZSOS+"_insseg";

    ISTR_LANG_WORD_DATA_LSCT = lang+"_lsct";
    if (lsct_spacing is None):
        lsct_spacing=[0];
    mpath_istr = os.path.join(pcfg.data_root, "dicts_v3/istr/dab_istr_" + lang);
    mprof_istr = neko_meta_stream_profile.get_default_training_loader_noto(
        ISTR_LANG_META, mpath_istr, ISTR_LANG_WORD_DATA, mpath_istr, ISTR_LANG_SINFO_NOTO_SIG,capacity=capacity);
    tome.mount_im_meta_stream(ISTR_LANG_META, mprof_istr, FPBPG);
    if (lsct):
        if (lsct == True or lsct == "lsct_repeater"):
            tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT,
                                       neko_lsct_ho_repeater_template.get_default_lsct_repeater(
                                           ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz,
                                           lsct_spacing), FPBPG);
        elif (lsct == "lsct_shuff"):
            tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT, neko_lsct_ho_shuff_template.get_default_lsct_shuff(
                ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz, lsct_spacing), FPBPG);
        else:
            fatal("unknown lsct type");
        if (lsct_sem):
            tome.add_simple_ossemseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                          ISTR_LSCT_LANG_TSK_INSSEG, None, FPBPG);
        if (lsct_ins):
            tome.add_ordered_cainsseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                           ISTR_LSCT_LANG_TSK_SEMSEG, None, FPBPG)
        tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                   ISTR_LSCT_LANG_TSK_ZSOS, None, FPBPG, maxT=32);
    dprofile_mjst = neko_datastream_profile.get_default_training_im_seq_loader(
        ISTR_LANG_WORD_DATA, pcfg.data_root, get_istr_zsl_bengali_gujarati("train", pcfg.data_root),
        ISTR_LANG_WORD_DATA_SIG, batchsize=bsf * 16, maxT=32);
    tome.mount_word_datastream_im_seq(ISTR_LANG_WORD_DATA, dprofile_mjst, FPBPG);

    tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA][0], tome.meta_eps[ISTR_LANG_META][0],
                               ISTR_LANG_TSK_ZSOS, None, FPBPG, maxT=32);

    return tome;
def arm_istr_zslc_bengali_gujarati_tr_sep_pun_tam(pcfg: neko_platform_cfg, tome: neko_320_tome, bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False):
    lang = "notbg"
    ISTR_LANG_META = "meta-" + lang;
    ISTR_LANG_WORD_DATA = lang;
    ISTR_LANG_WORD_DATA_SIG = "istr_" + lang;
    ISTR_LANG_TSK_ZSOS = lang + "_oszsl";
    ISTR_LANG_SINFO_NOTO_SIG = "noto_" + lang;
    FPBPG = "istr" + lang;
    ISTR_LSCT_LANG_TSK_ZSOS=ISTR_LANG_TSK_ZSOS+"_lsct";
    ISTR_LSCT_LANG_TSK_SEMSEG=ISTR_LSCT_LANG_TSK_ZSOS+"_semseg";
    ISTR_LSCT_LANG_TSK_INSSEG=ISTR_LSCT_LANG_TSK_ZSOS+"_insseg";

    ISTR_LANG_WORD_DATA_LSCT = lang+"_lsct";
    if (lsct_spacing is None):
        lsct_spacing=[0];
    mpath_istr = os.path.join(pcfg.data_root, "dicts_v3/istr/dab_istr_" + lang);
    mprof_istr = neko_meta_stream_profile.get_default_training_loader_noto(
        ISTR_LANG_META, mpath_istr, ISTR_LANG_WORD_DATA, mpath_istr, ISTR_LANG_SINFO_NOTO_SIG);
    tome.mount_im_meta_stream(ISTR_LANG_META, mprof_istr, FPBPG);
    if (lsct):
        if (lsct == True or lsct == "lsct_repeater"):
            tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT,
                                       neko_lsct_ho_repeater_template.get_default_lsct_repeater(
                                           ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz,
                                           lsct_spacing), FPBPG);
        elif (lsct == "lsct_shuff"):
            tome.mount_lsct_seq_stream(ISTR_LANG_WORD_DATA_LSCT, neko_lsct_ho_shuff_template.get_default_lsct_shuff(
                ISTR_LANG_WORD_DATA, pcfg.data_root, ISTR_LANG_WORD_DATA_LSCT, lsct_fsz, lsct_spacing), FPBPG);
        else:
            fatal("unknown lsct type");
        if (lsct_sem):
            tome.add_simple_ossemseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                          ISTR_LSCT_LANG_TSK_INSSEG, None, FPBPG);
        if (lsct_ins):
            tome.add_ordered_cainsseg_task(tome.feat_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                           ISTR_LSCT_LANG_TSK_SEMSEG, None, FPBPG)
        tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA_LSCT][0], tome.meta_eps[ISTR_LANG_META][0],
                                   ISTR_LSCT_LANG_TSK_ZSOS, None, FPBPG, maxT=32);
    dprofile_mjst = neko_datastream_profile.get_default_training_im_seq_loader(
        ISTR_LANG_WORD_DATA, pcfg.data_root, get_istr_zsl_bengali_gujarati_no_pt("train", pcfg.data_root),
        ISTR_LANG_WORD_DATA_SIG, batchsize=bsf * 16, maxT=32);
    tome.mount_word_datastream_im_seq(ISTR_LANG_WORD_DATA, dprofile_mjst, FPBPG);

    tome.add_simple_osseq_task(tome.seq_roi_eps[ISTR_LANG_WORD_DATA][0], tome.meta_eps[ISTR_LANG_META][0],
                               ISTR_LANG_TSK_ZSOS, None, FPBPG, maxT=32);

    return tome;
def arm_istr_zslc_bengali_gujarati_no_pun_tam(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2):
    tome= arm_istr_zslc_bengali_gujarati_tr(pcfg,tome,bsf);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome

def arm_istr_zslc_bengali_gujarati_no_pun_tam_fix(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False):
    tome= arm_istr_zslc_bengali_gujarati_tr_sep_pun_tam(pcfg,tome,bsf,lsct, lsct_fsz, lsct_spacing,lsct_sem=lsct_sem,lsct_ins=lsct_ins);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome
def arm_istr_zslc_bengali_gujarati(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2):
    tome= arm_istr_zslc_bengali_gujarati_tr(pcfg,tome,bsf);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "punjabi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "tamil","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome
def arm_istr_zslc_bengali_gujarati_fix(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False):
    tome= arm_istr_zslc_bengali_gujarati_tr_2(pcfg,tome,bsf,lsct, lsct_fsz, lsct_spacing, lsct_sem, lsct_ins);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "punjabi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "tamil","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome
def arm_istr_zslc_bengali_gujarati_fix_XC(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False,capacity=512):
    tome= arm_istr_zslc_bengali_gujarati_tr_2(pcfg,tome,bsf,lsct, lsct_fsz, lsct_spacing, lsct_sem, lsct_ins,capacity=capacity);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "punjabi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "tamil","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome
def arm_istr_zslc_bengali_gujarati_ptg(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None,lsct_sem=False,lsct_ins=False):
    tome= arm_istr_zslc_bengali_gujarati_tr_2(pcfg,tome,bsf,lsct, lsct_fsz, lsct_spacing, lsct_sem, lsct_ins);
    tome=arm_ist_test_task(pcfg,tome,"bengali","notbg");
    tome = arm_ist_test_task(pcfg, tome, "gujarati","notbg");
    tome = arm_ist_test_task(pcfg, tome, "hindi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "kannada","notbg");
    tome = arm_ist_test_task(pcfg, tome, "malayalam","notbg");
    tome = arm_ist_test_task(pcfg, tome, "marathi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "punjabi","notbg");
    tome = arm_ist_test_task(pcfg, tome, "tamil","notbg");
    tome = arm_ist_test_task(pcfg, tome, "telugu","notbg");
    tome = arm_ist_test_task(pcfg, tome, "odia","notbg");
    return tome


def arm_istr_affi_1tr(pcfg:neko_platform_cfg,tome:neko_320_tome,lang,bsf=2,lsct=False,lsct_fsz=64,lsct_spacing=None):
    tome= arm_ist_task_grp(pcfg,tome,lang,True,False,bsf,lsct, lsct_fsz, lsct_spacing);
    tome = arm_ist_test_task(pcfg,tome,"bengali",lang);
    tome = arm_ist_test_task(pcfg, tome, "gujarati",lang);
    tome = arm_ist_test_task(pcfg, tome, "hindi",lang);
    tome = arm_ist_test_task(pcfg, tome, "kannada",lang);
    tome = arm_ist_test_task(pcfg, tome, "malayalam",lang);
    tome = arm_ist_test_task(pcfg, tome, "marathi",lang);
    tome = arm_ist_test_task(pcfg, tome, "punjabi",lang);
    tome = arm_ist_test_task(pcfg, tome, "tamil",lang);
    tome = arm_ist_test_task(pcfg, tome, "telugu",lang);
    tome = arm_ist_test_task(pcfg, tome, "odia",lang);
    return tome
