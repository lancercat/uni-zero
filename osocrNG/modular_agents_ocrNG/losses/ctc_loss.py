import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf

class ctc_loss_agent_mk2(neko_module_wrapping_agent):
    INPUT_ctc_label_name = "ctc_label_name";
    INPUT_ctc_logit_name = "ctc_logit_name";
    INPUT_gt_length_name="gt_length";

    OUTPUT_ctc_loss_name = "loss_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.ctc_label_name = this.register_input(this.INPUT_ctc_label_name, iocvt_dict);
        this.ctc_logit_name = this.register_input(this.INPUT_ctc_logit_name, iocvt_dict);
        this.gt_length_name = this.register_input(this.INPUT_gt_length_name, iocvt_dict);
        this.ctc_loss_name = this.register_output(this.OUTPUT_ctc_loss_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ctc_label = workspace.get(this.ctc_label_name);
        ctc_logit = workspace.get(this.ctc_logit_name);
        gt_length = workspace.get(this.gt_length_name);
        predlen=torch.zeros_like(gt_length)+ctc_logit.shape[1];
        # CTC goes  T,N,C.  and needs log_probability
        logp=ctc_logit.squeeze(2).permute(1,0,2).log_softmax(2);
        l=trnf.ctc_loss(logp, ctc_label,predlen, gt_length , blank=0, reduction='none',
                      zero_infinity=True)/gt_length;
        workspace.add(this.ctc_loss_name,l);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   ctc_label_name, ctc_logit_name, gt_length_name,
                   ctc_loss_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_ctc_label_name: ctc_label_name, cls.INPUT_ctc_logit_name: ctc_logit_name,
                           cls.INPUT_gt_length_name: gt_length_name, cls.OUTPUT_ctc_loss_name: ctc_loss_name},
            "modcvt_dict": {}}}


def get_ctc_loss_agent_mk2(
        ctc_label_name, ctc_logit_name, gt_length_name,
        ctc_loss_name
):
    engine = ctc_loss_agent_mk2;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_ctc_label_name: ctc_label_name, engine.INPUT_ctc_logit_name: ctc_logit_name,
                       engine.INPUT_gt_length_name: gt_length_name, engine.OUTPUT_ctc_loss_name: ctc_loss_name},
        "modcvt_dict": {}}}

None




if __name__ == '__main__':
    print(ctc_loss_agent_mk2.print_default_setup_scripts())
    #