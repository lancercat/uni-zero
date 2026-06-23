

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_sdk.neko_framework_NG.names import P
from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.lsct_stream_profiles import neko_lsct_char_random_template

import os

from osocrNG.data_utils.common_data_presets_mk5.data.presets.protocol.train_data_dict_collection import \
    get_ctw_yolox_tr,get_ctw_yolox_val,get_ctw_yolox_te,get_enword_te
from osocrNG.data_utils.common_data_presets_mk5.data.presets.protocol.train_data_dict_collection import get_abi_mjst

from neko_sdk.ocr_modules.charset.testing_csets import jpnhv,jpnhv_kr_collateral
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.charset.kr_charset import contains_hangul
from neko_sdk.ocr_modules.charset.yi import yichars
from neko_sdk.log import fatal,info,warn
import torch

class neko_lsct_charset_jpn_nomltnew_kr_none:
    @classmethod
    def get_default_lsct_rand_charset(cls,data_root):
        fidx=torch.load(os.path.join(data_root, "synth_lsct/glyphdb_utf/filtered_index.pt"),weights_only=False);

        full_cset=fidx.keys();
        blklst=(set(jpnhv).difference(t1_3755)).difference(latin62);
        allowed=[];
        for t in full_cset:
            if(t in blklst):
                continue;
            if(contains_hangul(t)):
                continue;
            allowed.append(t);
        info("armed", str(len(allowed)),"characters");
        return allowed;
    @classmethod
    def get_default_lsctXL_rand_charset(cls,data_root,xtra_blklist=None):
        fidx=torch.load(os.path.join(data_root, "synth_lsct/glyphdb_utf_260202/filtered_index.pt"),weights_only=False);
        full_cset=fidx.keys();
        if(xtra_blklist is None):
            xtra_blklist=set();
        blklst=(set(jpnhv).union(jpnhv_kr_collateral).union(xtra_blklist).difference(t1_3755)).difference(latin62);
        allowed=[];
        for t in full_cset:
            if(t in blklst):
                continue;
            if(contains_hangul(t)):
                continue;
            allowed.append(t);
        info("armed", str(len(allowed)),"characters");
        return allowed;

from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome

# keep inductive
def arm_lsctc_no_jpnmlt_no_kr_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=8,lsct_fsz=32):
    CHS_LSCT_TSK_ZSOS="ctw_lsct_oszsl";
    LSCT = "LSCT_RND";
    FPBPG="freeform_lsct";
    batchsize=bsf*8;
    charset=neko_lsct_charset_jpn_nomltnew_kr_none.get_default_lsct_rand_charset(pcfg.data_root);
    profile=neko_lsct_char_random_template.get_default_lsct_rand_char(charset, pcfg.data_root, LSCT,batchsize);
    tome.mount_lsct_char_stream(LSCT, profile, FPBPG);
    tome.add_simple_oscr_task(tome.item_roi_eps[LSCT][0], tome.meta_eps[LSCT][0], CHS_LSCT_TSK_ZSOS, None,
                              FPBPG);
    return tome;
# keep inductive
def arm_lsctcXL_no_jpnmlt_no_kr_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=8,lsct_fsz=32):
    CHS_LSCT_TSK_ZSOS="ctw_lsct_oszsl";
    LSCT = "LSCT_RND";
    FPBPG="freeform_lsct";
    batchsize=bsf*8;
    charset=neko_lsct_charset_jpn_nomltnew_kr_none.get_default_lsctXL_rand_charset(pcfg.data_root);
    profile=neko_lsct_char_random_template.get_default_lsct_rand_charXL(charset, pcfg.data_root, LSCT,batchsize);
    tome.mount_lsct_char_stream(LSCT, profile, FPBPG);
    tome.add_simple_oscr_task(tome.item_roi_eps[LSCT][0], tome.meta_eps[LSCT][0], CHS_LSCT_TSK_ZSOS, None,
                              FPBPG);
    return tome;
def arm_lsctcXL_no_jpnmlt_no_kr_no_yi_task_grp(pcfg:neko_platform_cfg,tome:neko_320_tome,bsf=8,lsct_fsz=32):
    CHS_LSCT_TSK_ZSOS="ctw_lsct_oszsl";
    LSCT = "LSCT_RND";
    FPBPG="freeform_lsct";
    batchsize=bsf*8;
    charset=neko_lsct_charset_jpn_nomltnew_kr_none.get_default_lsctXL_rand_charset(pcfg.data_root,xtra_blklist=set(yichars));
    profile=neko_lsct_char_random_template.get_default_lsct_rand_charXL(charset, pcfg.data_root, LSCT,batchsize,fsz=lsct_fsz);
    tome.mount_lsct_char_stream(LSCT, profile, FPBPG);
    tome.add_simple_oscr_task(tome.item_roi_eps[LSCT][0], tome.meta_eps[LSCT][0], CHS_LSCT_TSK_ZSOS, None,
                              FPBPG);
    return tome;
