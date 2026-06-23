import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg;
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
# lets say if we have ideal reward, can the PPO learn?
class neko_fake_fpbp_loss_anchor_name_matching(neko_module_wrapping_agent):
    PARAM_expected_anchor_name="expected_anchor_name";
    INPUT_preferred_anchor_name="preferred_anchor_name";
    OUTPUT_fake_per_instance_loss="fake_per_instance_loss";
    PARAM_loss_device="loss_device";
    def set_etc(this,param):
        this.expected_anchor_name=neko_get_arg(this.PARAM_expected_anchor_name,param);
        this.loss_dev=neko_get_arg(this.PARAM_loss_device,param,"cuda:0");
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.preferred_anchors=this.register_input(this.INPUT_preferred_anchor_name,iocvt_dict);
        this.perinst_loss=this.register_output(this.OUTPUT_fake_per_instance_loss,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        preferred_anchors=workspace.get(this.preferred_anchors);
        loss=[];
        for a in preferred_anchors:
            if(a==this.expected_anchor_name):
                loss.append(0);
            else:
                loss.append(1);
        loss=torch.tensor(loss,device=this.loss_dev,dtype=torch.float32);
        workspace.add(this.perinst_loss,loss);
        return workspace;
def get_neko_fake_fpbp_loss_anchor_name_matching(preferred_anchor_name,
    fake_per_instance_loss_name,expected_anchor_name,loss_device="cuda:0"):
    engine = neko_fake_fpbp_loss_anchor_name_matching;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_preferred_anchor_name: preferred_anchor_name, engine.OUTPUT_fake_per_instance_loss: fake_per_instance_loss_name}, engine.PARAM_expected_anchor_name: expected_anchor_name, engine.PARAM_loss_device: loss_device, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_fake_fpbp_loss_anchor_name_matching.get_default_configuration_scripts());