from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.ocr import arm_chs_jpnkr_hori_task_grp, arm_egptology_task_grp,arm_ctwch_task_grp,arm_mjst_task_grp
#from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.panseg import arm_cityscape_task_grp
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.headless_lsct import arm_lsctcXL_no_jpnmlt_no_kr_task_grp, \
    arm_lsctcXL_no_jpnmlt_no_kr_no_yi_task_grp
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.istr_lst import arm_ist_task_grp,arm_istr_zslc_bengali_gujarati_no_pun_tam
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.istr_lst import arm_ist_task_grp, \
    arm_istr_zslc_bengali_gujarati_no_pun_tam, arm_istr_zslc_bengali_gujarati_no_pun_tam_fix
def arm_taskgrp(pcfg,tome):
    # tome=arm_ctwch_task_grp(pcfg,tome,True);
    # tome = arm_egptology_task_grp(pcfg, tome);
    # tome=arm_mjst_task_grp(pcfg,tome);
    tome=arm_lsctcXL_no_jpnmlt_no_kr_no_yi_task_grp(pcfg,tome,lsct_fsz=64,bsf=1);
    tome = arm_ist_task_grp(pcfg, tome, "punjabi", True, True,2);
    tome = arm_ist_task_grp(pcfg, tome, "tamil", True, True,2);
    tome =arm_istr_zslc_bengali_gujarati_no_pun_tam_fix(pcfg,tome,lsct="lsct_shuff",lsct_ins=False);
    # tome=arm_cityscape_task_grp(pcfg,tome);
    return tome;
