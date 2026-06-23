

import glob
import os.path
import sys

from loadout import modf,agtf,datf,acfg,trdcfg
from neko_sdk.neko_framework_NG.UAE.neko_mission_agent import neko_test_mission_agent_single_im

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.log import fatal,info,warn

if __name__ == '__main__':
    DQN="dq";
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    cfg.arm_wandb(project="watch_and_control");
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
    modset.load("_E2");
    modset.eval_mode();
    # modset.bfloat16();

    e=neko_environment(assets_dict={},queue_dict={},modset=modset);

    src="./athenaNG-310/";
    dst="./athenaNG_results/";

    tac=af.get_imbased_tester(anchors,dst,"");
    tester=neko_test_mission_agent_single_im(tac["params"]);
    info("starting");

    als=[os.path.basename(i) for i in glob.glob(os.path.join(src,"*"))];
    for l in als:
        tester.arm_path(src,dst,l,"png");
        tester.take_action(neko_workspace(),e);
    pass;

