
# in mk2, labels are computed on the fly---
# we are not going to save this 10ms of latency.
# when time calls, this can be an async call.

import torch
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.OSR.classification.agents.neko_abstract_mklabel import neko_abstract_label_maker,osr_translate
class neko_ccd_label_making_agent_mk2(neko_abstract_label_maker):
    OUTPUT_flatten_label="flatten_tensor_label";
    OUTPUT_tensor_gt_length="tensor_gt_length";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.flatten_label = this.register_output(this.OUTPUT_flatten_label, iocvt_dict);
        this.tensor_gt_length = this.register_output(this.OUTPUT_tensor_gt_length, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev_indicator = workspace.get(this.dev_indicator);
        sam_utf = workspace.get(this.sam_utf);
        tdict = workspace.get(this.tdict);
        rawlab, lens = osr_translate.translate(sam_utf, tdict);
        tenlabel=osr_translate.flatten_labels(rawlab,dev_indicator.device);
        tenlen=torch.tensor(lens,device=dev_indicator.device,dtype=torch.long);
        workspace.add(this.flatten_label,tenlabel);
        workspace.add(this.tensor_gt_length,tenlen);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   dev_indicator, sam_utf, tdict,
                   flatten_label, tensor_gt_length
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_dev_indicator: dev_indicator, cls.INPUT_sam_utf: sam_utf, cls.INPUT_tdict: tdict,
                           cls.OUTPUT_flatten_label: flatten_label, cls.OUTPUT_tensor_gt_length: tensor_gt_length},
            "modcvt_dict": {}}}

class neko_ccd_label_making_agent_mk2s(neko_ccd_label_making_agent_mk2):

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev_indicator = workspace.get(this.dev_indicator);
        sam_utf = workspace.get(this.sam_utf);
        tdict = workspace.get(this.tdict);
        rawlab, lens = osr_translate.translate_strict(sam_utf, tdict);
        tenlabel=osr_translate.flatten_labels(rawlab,dev_indicator.device);
        tenlen=torch.tensor(lens,device=dev_indicator.device,dtype=torch.long);
        workspace.add(this.flatten_label,tenlabel);
        workspace.add(this.tensor_gt_length,tenlen);
        return workspace, environment;
if __name__ == '__main__':
    neko_ccd_label_making_agent_mk2.print_default_setup_scripts()