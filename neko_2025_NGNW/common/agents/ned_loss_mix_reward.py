from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

import torch

class neko_ned_thresholding_mix_agent_32x(neko_module_wrapping_agent):

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
        ned = torch.tensor(workspace.get(this.ned),dtype=detached_loss.dtype,device=detached_loss.device);
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
class neko_loss_agent_32x(neko_module_wrapping_agent):

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
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        detached_loss = workspace.get(this.loss).detach();
        workspace.add(this.penalty,detached_loss);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   loss, ned,
                   penalty
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_loss: loss, cls.INPUT_ned: ned, cls.OUTPUT_penalty: penalty}, "modcvt_dict": {}}}
