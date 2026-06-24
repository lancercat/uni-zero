import shutil
# Standard Library and Third-Party

# Project Configuration and Utilities
import sys

import torch.backends.cudnn
import os
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome,neko_320_net_size_meta
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.egptv3 import arm_egptology_v3R_task_grp


from net import get_tea

import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method("fork");
    torch.backends.cudnn.benchmark=False; # bcs cotrianing will mess it up.
    tome = neko_320_tome();
    tome.net_size_meta=neko_320_net_size_meta({
        neko_320_net_size_meta.PARAM_WORD_IM_DATA_SIZE: [32,128],
        neko_320_net_size_meta.PARAM_ITEM_IM_DATA_SIZE: [48,48]
    });
    if(len(sys.argv)>1):
        pcfg=neko_platform_cfg(sys.argv[1]);
    else:
        pcfg=neko_platform_cfg(None);
    tome = arm_egptology_v3R_task_grp(pcfg, tome);
    curname=os.path.basename(os.getcwd());
    ITRS ={"_E1_I20000":80000, "_E1_I40000":100000, "_E2":120000};
    # with torch.autocast(device_type=pcfg.devices[0], dtype=torch.float16, enabled=True):
    ars={};
    for iterk in ITRS:
        logpath = os.path.join("/run/media/lasercat/320-eccv/results/hydra_results/", curname+iterk);
        shutil.rmtree(logpath, ignore_errors=True);
        os.makedirs(logpath)
        te = get_tea(pcfg, tome, 0, 0, logpath);
        te.modset.load(iterk);
        rds=te.test(0,ITRS[iterk]);
        ars[iterk]=rds;
        pass;

    torch.save(ars,os.path.join("/run/media/lasercat/320-eccv/results/hydra_results/", curname+".pt"));
