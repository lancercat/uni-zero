import torch

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNK

from neko_sdk.OSR.classification.agents.neko_abstract_mklabel import neko_abstract_label_maker,osr_translate
class neko_label_translation_as_is_flattened(neko_abstract_label_maker):
    OUTPUT_gt_tensor="gt_tensor"; # the exact ground truth depends on the type, tho.

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.gt_tensor = this.register_output(this.OUTPUT_gt_tensor, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev_indicator = workspace.get(this.dev_indicator);
        sam_utf = workspace.get(this.sam_utf);
        tdict = workspace.get(this.tdict);
        aids,l=osr_translate.translate(sam_utf,tdict);
        workspace.add(this.gt_tensor,osr_translate.translate(aids,dev_indicator.device));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   dev_indicator, sam_utf, tdict,
                   gt_tensor
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_dev_indicator: dev_indicator, cls.INPUT_sam_utf: sam_utf, cls.INPUT_tdict: tdict,
                           cls.OUTPUT_gt_tensor: gt_tensor}, "modcvt_dict": {}}}

