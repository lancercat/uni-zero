import copy
import sys

from loadout import modf,agtf,datf,acfg,trdcfg

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.log import fatal,info,warn
import os
if __name__ == '__main__':
    DQN="dq";
    if(len(sys.argv)>2):
        cfg=neko_platform_cfg(sys.argv[2]);
    else:
        cfg=neko_platform_cfg(None);
    # cfg.save_root=os.path.join("/run/media/lasercat/writebuffer/310-2/",os.path.basename(os.getcwd()));


    cfg.arm_wandb(project="watch_and_control");
    mf=modf(cfg);
    af=agtf(cfg);
    anchors = acfg();
    data_partitions=trdcfg();
    modcfgdict, bogo_dict = {}, {};
    trdf=trdcfg(cfg);
    trad, trqd, trm, tedd=datf.get_mk3_benchmark_release(cfg.data_root,trdf,DQN);
    modcfgdict, bogo_dict = mf.config_for_training(modcfgdict, bogo_dict, anchors, trm);
    modset = neko_module_opt_setNG();
    modset.arm_modules(modcfgdict, bogo_dict);
    modset.to(cfg.devices[0]);
    # modset.bfloat16();


    e=neko_environment(assets_dict={},queue_dict=trqd,modset=modset);

    ITRKs=["_E2"];
    vtedd=copy.copy(tedd);
    for dsk in tedd["tests"]:
        for itrk in ITRKs:
            vtedd["tests"]={dsk:tedd["tests"][dsk]};
            modset.load(itrk);
            tac=af.get_debuggers({"main":tedd},anchors,os.path.join(cfg.log_root,dsk+itrk),None,"",False,9);
            tra=af.get_trainer(tac,anchors,DQN);
            info("starting");
            tra.mount_environment({},e);
            tra.eval(16);
            tra.dumpcache(os.path.join(cfg.log_root,dsk+itrk,"proto.pt"));

    pass;

