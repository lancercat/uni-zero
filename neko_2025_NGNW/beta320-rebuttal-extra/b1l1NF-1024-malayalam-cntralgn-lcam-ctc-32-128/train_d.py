# Standard Library and Third-Party

# Project Configuration and Utilities
import sys

import torch.backends.cudnn

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.templates.tome import neko_320_tome,neko_320_net_size_meta

from taskgrp import arm_taskgrp
from net import get_tra

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=False; # bcs cotrianing will mess it up.
    tome = neko_320_tome();
    tome.net_size_meta=neko_320_net_size_meta({
        neko_320_net_size_meta.PARAM_WORD_IM_DATA_SIZE: [32,128],
    });
    if(len(sys.argv)>1):
        pcfg=neko_platform_cfg(sys.argv[1]);
    else:
        pcfg=neko_platform_cfg(None);
    tome=arm_taskgrp(pcfg,tome);
    tr=get_tra(pcfg, tome,60000,6);
    # tr.modset.load("_E2");
    tr.train(True);


