from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import numpy as np
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_tensor_im_visualization(neko_module_wrapping_agent):
    INPUT_tensor_img = "pred_text";
    OUTPUT_raw_img_vis="padded_raw_im_vis";
    PARAM_mean="mean";
    PARAM_var="var";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tensor_img = this.register_input(this.INPUT_tensor_img, iocvt_dict);
        this.raw_img_vis = this.register_output(this.OUTPUT_raw_img_vis, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.mean = neko_get_arg(this.PARAM_mean, params,127);
        this.var = neko_get_arg(this.PARAM_var, params,127);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tensor_img = workspace.get(this.tensor_img);
        img = [[i] for i in (tensor_img.detach() * this.var + this.mean).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)];
        workspace.add(this.raw_img_vis,img);
        return workspace, environment;

def get_neko_tensor_im_visualization(tensor_img,raw_img_vis,mean, var):
    engine = neko_tensor_im_visualization;
    return {"agent": engine,
            "params": {"iocvt_dict": {engine.INPUT_tensor_img: tensor_img, engine.OUTPUT_raw_img_vis: raw_img_vis},
                       engine.PARAM_mean: mean, engine.PARAM_var: var, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_tensor_im_visualization.print_default_setup_scripts();
