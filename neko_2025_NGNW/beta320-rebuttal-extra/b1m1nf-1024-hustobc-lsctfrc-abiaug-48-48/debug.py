
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


    curname=os.path.basename(os.getcwd());


    ars=torch.load(os.path.join("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", curname + ".pt"));

    print(ars)