
# Standard Library and Third-Party

# Project Configuration and Utilities
import sys

import torch.backends.cudnn

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome,neko_320_net_size_meta
from neko_2025_NGNW.common.object_32x_presets.tasks.taskgrps.ostr_xtest import arm_chs_jpnkr_hori_task_grp_te_xra


from net import get_tea

import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method("fork");
    torch.backends.cudnn.benchmark=False; # bcs cotrianing will mess it up.
    tome = neko_320_tome();
    tome.net_size_meta=neko_320_net_size_meta({
        neko_320_net_size_meta.PARAM_WORD_IM_DATA_SIZE: [32,128]
    });
    if(len(sys.argv)>1):
        pcfg=neko_platform_cfg(sys.argv[1]);
    else:
        pcfg=neko_platform_cfg(None);
    tome = arm_chs_jpnkr_hori_task_grp_te_xra(pcfg, tome);

    te=get_tea(pcfg, tome,0,0,"/run/media/lasercat/320-eccv/results/hydra_results_eccv_demo_uni/");

    # with torch.autocast(device_type=pcfg.devices[0], dtype=torch.float16, enabled=True):
    te.modset.load("_E2");
    te.test(9,9);


