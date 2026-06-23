from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNKREP

import torch

class translate_gt_tokens_list_agent(neko_module_wrapping_agent):
    INPUT_gt_tokens_list="GT";
    INPUT_tdict="tdict";
    OUTPUT_gt_tokens_list_wunk= "GT_wunk"; # damn, if we use un-unked text to compute reward we are nuking ourselves.

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.gt_tokens_list = this.register_input(this.INPUT_gt_tokens_list, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.gt_tokens_list_wunk = this.register_output(this.OUTPUT_gt_tokens_list_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt_tokens_list = workspace.get(this.gt_tokens_list);
        tdict = workspace.get(this.tdict);
        wunk=[[ c if c in tdict else tUNKREP for c in gt_tokens]for gt_tokens in gt_tokens_list];
        workspace.add(this.gt_tokens_list_wunk,wunk);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   gt_tokens_list, tdict,
                   gt_tokens_list_wunk
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_gt_tokens_list: gt_tokens_list, cls.INPUT_tdict: tdict, cls.OUTPUT_gt_tokens_list_wunk: gt_tokens_list_wunk},
            "modcvt_dict": {}}}

def get_translate_gt_agent(
        gt_tokens_list, tdict,
        gt_tokens_list_wunk
):
    engine = translate_gt_tokens_list_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_gt_tokens_list: gt_tokens_list, engine.INPUT_tdict: tdict,
                                                       engine.OUTPUT_gt_tokens_list_wunk: gt_tokens_list_wunk},
                                        "modcvt_dict": {}}}
