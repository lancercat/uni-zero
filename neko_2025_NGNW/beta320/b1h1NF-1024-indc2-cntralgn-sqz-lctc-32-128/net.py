# Standard Library and Third-Party

# Project Configuration and Utilities
import sys

from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_mod_factory, global_mod_cfg
from neko_2025_NGNW.common.factories.presets.neko_320_netbuilder import neko_object320_oscr_no_route_norm_feat_sqz_ctc_lsctrest_sem_corrw05
from neko_2025_NGNW.common.agents.training_and_testing.neko_320_training import neko_training_agent_320
from neko_2025_NGNW.common.object_32x_presets.base_loader_factory import (
    neko_32x_training_data_loader_agent_factory,neko_32x_testing_meta_loader_agent_factory)
def arm_tedaf_totome(pcfg,tome):
    tedaf = neko_32x_testing_meta_loader_agent_factory(pcfg.data_root);
    #
    for tk in tome.imcls_testing_tasks:
        te_dacfg, te_metaldr_cfg, te_sinfo_cfg = tedaf.get_testing_bundle(
            tome.imcls_testing_tasks[tk].data_ldr_profile,
            tome.imcls_testing_tasks[tk].meta_ldr_profile, pcfg);  # defines testing data loader agents.
        tome.arm_testing_data(tome.imcls_testing_tasks[tk], te_dacfg, te_metaldr_cfg, te_sinfo_cfg);

    for tk in tome.im_seqcls_testing_tasks:
        te_dacfg, te_metaldr_cfg, te_sinfo_cfg = tedaf.get_testing_bundle(
            tome.im_seqcls_testing_tasks[tk].data_ldr_profile,
            tome.im_seqcls_testing_tasks[tk].meta_ldr_profile, pcfg);  # defines testing data loader agents.
        tome.arm_testing_data(tome.im_seqcls_testing_tasks[tk], te_dacfg, te_metaldr_cfg, te_sinfo_cfg);
    return tome

def get_tra(pcfg,tome,vitr=100000,vepo=5):
    tome=arm_tedaf_totome(pcfg, tome);
    nf = neko_object320_oscr_no_route_norm_feat_sqz_ctc_lsctrest_sem_corrw05(pcfg, {
        global_mod_cfg.PARAM_BN_HAS_AFFINE: True,
        global_mod_cfg.PARAM_FEAT_DIM: 1024  # according to https://arxiv.org/abs/2506.04807 we boost it

    });
    modcfg, bogocfg = virtual_mod_factory.empty_modcfgs();
    modcfg, bogocfg = nf.arm_taskgrp_mods(modcfg, bogocfg, tome);
    modset = neko_module_opt_setNG();
    modset.arm_modules(modcfg[virtual_mod_factory.KEY_modules], bogocfg);
    modset.to(pcfg.devices[0]);

    daf = neko_32x_training_data_loader_agent_factory(pcfg.data_root);
    dacfg = daf.get_ldr_agent(tome.diskldr_dprofile_dict, tome.diskldr_mprofile_dict, tome.diskldr_lprofile_dict, pcfg);
    trad = {};
    tome.save_each = 20000;
    for a in tome.fpbpgrps:
        trad[a] = nf.get_training_task_fpbp_grp(tome.fpbpgrps[a]);
        pass;
    teacfgs = nf.get_testing_task(tome);
    tra = neko_training_agent_320({
        neko_training_agent_320.PARAM_MODSET: modset,
        neko_training_agent_320.PARAM_PCFG: pcfg,
        neko_training_agent_320.PARAM_FPBP_ACFG_DIC: trad,
        neko_training_agent_320.PARAM_DEVICES: pcfg.devices,
        neko_training_agent_320.PARAM_TEAGTS: [teacfgs],
        neko_training_agent_320.PARAM_CHK_EACH: 5000,
        neko_training_agent_320.PARAM_SAVE_EACH: tome.save_each,
        neko_training_agent_320.PARAM_VITR: vitr,
        neko_training_agent_320.PARAM_EPOC: vepo,
        neko_training_agent_320.PARAM_TR_LDR_ACFG: dacfg
    });
    return tra;
def get_tea(pcfg,tome,vitr=100000,vepo=5,logging_path=None):
    nf = neko_object320_oscr_no_route_norm_feat_sqz_ctc_lsctrest_sem_corrw05(pcfg, {
        global_mod_cfg.PARAM_BN_HAS_AFFINE: True,
        global_mod_cfg.PARAM_FEAT_DIM: 1024  # according to https://arxiv.org/abs/2506.04807 we boost it
    });
    modcfg, bogocfg = virtual_mod_factory.empty_modcfgs();
    modcfg, bogocfg = nf.arm_taskgrp_mods(modcfg, bogocfg, tome);
    modset = neko_module_opt_setNG();
    modset.arm_modules(modcfg[virtual_mod_factory.KEY_modules], bogocfg);
    modset.to(pcfg.devices[0]);
    tome=arm_tedaf_totome(pcfg, tome);
    dacfg=awa.empty();
    trad = {};
    tome.save_each = 20000;

    teacfgs = nf.get_testing_task(tome,logging_path);
    tra = neko_training_agent_320({
        neko_training_agent_320.PARAM_MODSET: modset,
        neko_training_agent_320.PARAM_PCFG: pcfg,
        neko_training_agent_320.PARAM_FPBP_ACFG_DIC: trad,
        neko_training_agent_320.PARAM_DEVICES: pcfg.devices,
        neko_training_agent_320.PARAM_TEAGTS: [teacfgs],
        neko_training_agent_320.PARAM_CHK_EACH: 5000,
        neko_training_agent_320.PARAM_SAVE_EACH: tome.save_each,
        neko_training_agent_320.PARAM_VITR: vitr,
        neko_training_agent_320.PARAM_EPOC: vepo,
        neko_training_agent_320.PARAM_TR_LDR_ACFG: dacfg
    });
    return tra;

