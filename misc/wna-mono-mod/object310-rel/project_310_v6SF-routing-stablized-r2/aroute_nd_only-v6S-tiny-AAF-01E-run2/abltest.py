import sys

import torch

from loadout import modf,agtf,datf,acfg,trdcfg

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.log import fatal,info,warn
import copy
import os
if __name__ == '__main__':
    DQN="dq";
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    cfg.arm_wandb(project="watch_and_control");
    # cfg.save_root=os.path.join("/run/media/lasercat/writebuffer/310-2/",os.path.basename(os.getcwd()));

    mf=modf(cfg);
    af=agtf(cfg);
    trdf=trdcfg(cfg)
    anchors = acfg();
    modcfgdict, bogo_dict = {}, {};
    trad, trqd, trm, tedd=datf.get_mk3_benchmark_plus(cfg.data_root,trdf,DQN);

    modcfgdict, bogo_dict = mf.config_for_training(modcfgdict, bogo_dict, anchors, trm);
    modset = neko_module_opt_setNG();
    modset.arm_modules(modcfgdict, bogo_dict);
    modset.to(cfg.devices[0]);
    # modset.bfloat16();

    e=neko_environment(assets_dict={},queue_dict=trqd,modset=modset);
    ITRKs=["_E1_I20000","_E1_I40000","_E1_I60000","_E1_I80000","_E1_I100000",
               "_E1_I120000","_E1_I140000","_E1_I160000","_E1_I180000","_E2"];
    dsks=["JPNHV-JPNHV-GZSL"];
    vtedd=copy.copy(tedd);
    ard={};

    for dsk in dsks:
        ard[dsk]={};
        for itrk in ITRKs:
            vtedd["tests"]={dsk:tedd["tests"][dsk]};
            modset.load(itrk);
            tac=af.get_testers({"main":vtedd},anchors);
            tra=af.get_trainer(tac,anchors,DQN);
            info("starting");
            tra.mount_environment({},e);
            rd=tra.eval(16);
            ard[dsk][itrk]=rd;
            pass;
    torch.save(ard,"ablruns.pt");
    pass;

