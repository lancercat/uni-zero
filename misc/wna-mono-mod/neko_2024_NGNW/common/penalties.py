import torch

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
import editdistance
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_perinst_ned_agent(neko_module_wrapping_agent):

    INPUT_gt="gt";
    INPUT_pred="pred";
    INPUT_dev_ref="devref";
    OUTPUT_ned="ned";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.dev_ref = this.register_input(this.INPUT_dev_ref, iocvt_dict);
        this.gt = this.register_input(this.INPUT_gt, iocvt_dict);
        this.pred = this.register_input(this.INPUT_pred, iocvt_dict);
        this.ned = this.register_output(this.OUTPUT_ned, iocvt_dict);


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev_ref = workspace.get(this.dev_ref);
        gt = workspace.get(this.gt);
        pred = workspace.get(this.pred);
        aed=[];
        for p,g in zip(pred,gt):
            norm_ED=editdistance.eval(p.lower(), g.lower()) / len(g); # case insensitive annotation are used for training....
            aed.append(norm_ED);
        workspace.add(this.ned,torch.tensor(aed,dtype=torch.float32,device=dev_ref.device));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   dev_ref, gt, pred,
                   ned
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_dev_ref: dev_ref, cls.INPUT_gt: gt, cls.INPUT_pred: pred, cls.OUTPUT_ned: ned},
            "modcvt_dict": {}}}




class neko_ned_thresholding_mix_agent(neko_module_wrapping_agent):

    INPUT_ned="ned";
    INPUT_loss = "loss";
    OUTPUT_penalty = "penalty";

    PARAM_thresh="ned_thres"; # if the ned is too large to make sense (consider how the model will behave at the beginning) set it to a constvalue (to discourage the model from taking this option)
    PARAM_ned_inf="ned_inf"; # the said value, say 50

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.loss = this.register_input(this.INPUT_loss, iocvt_dict);
        this.ned = this.register_input(this.INPUT_ned, iocvt_dict);
        this.penalty = this.register_output(this.OUTPUT_penalty, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.ned_inf = neko_get_arg(this.PARAM_ned_inf, params,100);
        this.thresh = neko_get_arg(this.PARAM_thresh, params,0.6);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        detached_loss = workspace.get(this.loss).detach();
        ned = workspace.get(this.ned);
        workspace.add(this.penalty,torch.clamp(ned,0,this.thresh)*this.ned_inf+detached_loss);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   loss, ned,
                   penalty,
                   ned_inf, thresh
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_loss: loss, cls.INPUT_ned: ned, cls.OUTPUT_penalty: penalty},
            cls.PARAM_ned_inf: ned_inf, cls.PARAM_thresh: thresh, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_ned_thresholding_mix_agent.print_default_setup_scripts()