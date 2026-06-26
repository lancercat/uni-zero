
# Standard Library and Third-Party

# Project Configuration and Utilities
import sys

import torch.backends.cudnn

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome,neko_320_net_size_meta
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.ocr import arm_chs_jpnkr_hori_task_grp,arm_ctwch_task_grp,arm_mjst_task_grp
import json
from taskgrp import arm_taskgrp
from net import get_tra
import shutil
import os


from net import get_tea


import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method("fork");
    torch.backends.cudnn.benchmark=False; # bcs cotrianing will mess it up.
    tome = neko_320_tome();
    tome.net_size_meta=neko_320_net_size_meta({
        neko_320_net_size_meta.PARAM_WORD_IM_DATA_SIZE: [32,128]
    });
    if(len(sys.argv)>2):
        pcfg=neko_platform_cfg(sys.argv[1]);
        ITRS = json.loads(sys.argv[2]);
    else:
        ITRS = { "_E2": 120000};
        pcfg=neko_platform_cfg(None);
    tome=arm_taskgrp(pcfg,tome);

    curname=os.path.basename(os.getcwd());


    save_path = os.path.join("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", curname + ".pt");
    # --- CHECK IF PT FILE EXISTS ---
    if os.path.exists(save_path):
        print(f"Skipping: {save_path} already exists.")
        sys.exit(0)  # Or return if inside a function
    ars = {};
    for iterk in ITRS:
        logpath = os.path.join("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", curname + iterk);
        shutil.rmtree(logpath, ignore_errors=True);
        os.makedirs(logpath)
        te = get_tra(pcfg, tome, 0, 0);
        te.modset.load(iterk);
        rds = te.test(0, ITRS[iterk]);
        ars[iterk] = rds;
        pass;

    torch.save(ars, os.path.join("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", curname + ".pt"));

