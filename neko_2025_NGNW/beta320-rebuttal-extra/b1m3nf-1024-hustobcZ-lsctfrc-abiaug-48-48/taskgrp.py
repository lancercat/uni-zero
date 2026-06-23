from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.hustobc import arm_hustobc_zero_task_grp
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.headless_lsct import arm_lsctcXL_no_jpnmlt_no_kr_task_grp

def arm_taskgrp(pcfg,tome):
    # tome=arm_ctwch_task_grp(pcfg,tome,True);
    tome = arm_hustobc_zero_task_grp(pcfg, tome,"abinet");
    tome=arm_lsctcXL_no_jpnmlt_no_kr_task_grp(pcfg,tome,lsct_fsz=64,bsf=1);

    # tome=arm_mjst_task_grp(pcfg,tome);
    # tome=arm_chs_jpnkr_hori_task_grp(pcfg,tome,bsf=2,lsct=False,lsct_sem=False,lsct_ins=False,lsct_fsz=64);
    # tome=arm_cityscape_task_grp(pcfg,tome);
    return tome;
