import copy

# in mk2, labels are computed on the fly---
# we are not going to save this 10ms of latency.
# when time calls, this can be an async call.
from osocrNG.sptokens import tEOS,tBOS
import torch
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.OSR.classification.agents.neko_abstract_mklabel import neko_abstract_label_maker,osr_translate
class neko_rnn_label_making_agent_mk2(neko_abstract_label_maker):
    OUTPUT_flatten_label="flatten_tensor_label";
    OUTPUT_tensor_gt_length="tensor_gt_length";
    PARAM_count_eos="count_eos";
    PARAM_count_bos="count_bos";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.flatten_label = this.register_output(this.OUTPUT_flatten_label, iocvt_dict);
        this.tensor_gt_length = this.register_output(this.OUTPUT_tensor_gt_length, iocvt_dict);
        pass;

    # might worth it to double tap
    def set_etc(this, params):
        # [2504.02732] Why do LLMs attend to the first token?; is one enough :)
        this.bos=[tBOS for _ in range(len(neko_get_arg(this.PARAM_count_bos,params,0)))];
        this.eos=[tEOS for _ in range(len(neko_get_arg(this.PARAM_count_eos,params,1)))];
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev_indicator = workspace.get(this.dev_indicator);
        sam_utf = workspace.get(this.sam_utf);
        tdict = workspace.get(this.tdict);
        alt_utf=[this.bos+_+this.eos for _ in sam_utf]; # adding eos for RNN
        rawlab, lens = osr_translate.translate(alt_utf, tdict);
        tenlabel=osr_translate.flatten_labels(rawlab,dev_indicator.device);
        tenlen=torch.tensor(lens,device=dev_indicator,dtype=torch.long)+1; # add the EOS.
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

   
