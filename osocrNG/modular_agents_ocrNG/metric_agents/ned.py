import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import editdistance

class neko_ned_agent(neko_module_wrapping_agent):
    INPUT_pred_text="pred_text";
    INPUT_gt_text="gt_text";
    OUTPUT_ned="ned";
    PARAM_device="device";
    PARAM_dtype="dtype";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.gt_text = this.register_input(this.INPUT_gt_text, iocvt_dict);
        this.pred_text = this.register_input(this.INPUT_pred_text, iocvt_dict);
        this.ned = this.register_output(this.OUTPUT_ned, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.device = neko_get_arg(this.PARAM_device, params,"cpu");
        this.dtype = neko_get_arg(this.PARAM_dtype, params,torch.float32);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt_text = workspace.get(this.gt_text);
        pred_text = workspace.get(this.pred_text);
        ned=torch.tensor([editdistance.eval(pred_text[i], gt_text[i]) / len(gt_text[i]) for i in range(len(pred_text))],dtype=this.dtype,device=this.device);
        workspace.add(this.ned,ned);

        return workspace, environment;

def get_neko_ned_agent(
        gt_text, pred_text,
        ned,
        device="cpu", dtype=torch.float32
):
    engine = neko_ned_agent;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_gt_text: gt_text, engine.INPUT_pred_text: pred_text, engine.OUTPUT_ned: ned},
        engine.PARAM_device: device, engine.PARAM_dtype: dtype, "modcvt_dict": {}}}

# this will add an overall performance as a preference and also a random factor to reduce the chance of having a tie.
class neko_ned_agent_as_penalization(neko_ned_agent):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt_text = workspace.get(this.gt_text);
        pred_text = workspace.get(this.pred_text);
        ned=torch.tensor([editdistance.eval(pred_text[i], gt_text[i]) / len(gt_text[i]) for i in range(len(pred_text))],dtype=this.dtype,device=this.device);
        ned=ned+ned.mean()*0.001+torch.rand_like(ned)*0.000001; #so it always have an argmax
        workspace.add(this.ned,ned);

        return workspace, environment;
def get_neko_ned_agent_as_penalization(
        gt_text, pred_text,
        ned,
        device="cpu", dtype=torch.float32
):
    engine = neko_ned_agent_as_penalization;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_gt_text: gt_text, engine.INPUT_pred_text: pred_text, engine.OUTPUT_ned: ned},
        engine.PARAM_device: device, engine.PARAM_dtype: dtype, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_ned_agent.print_default_setup_scripts()





#



