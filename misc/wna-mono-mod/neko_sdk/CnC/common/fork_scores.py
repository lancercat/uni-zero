import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace


class neko_slice(neko_module_wrapping_agent):
    INPUT_integrated_tensor="tensor";
    OUTPUT_names="names";
    PARAM_dim="dim";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.integrated_tensor = this.register_input(this.INPUT_integrated_tensor, iocvt_dict);
        this.names = this.register_output_list(this.OUTPUT_names, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.dim = neko_get_arg(this.PARAM_dim, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        integrated_tensor = workspace.get(this.integrated_tensor);
        for i in range(len(this.names)):
            workspace.add(this.names[i],torch.select(integrated_tensor,this.dim,i))
        return workspace, environment;

def get_neko_slice(
        integrated_tensor,
        names,
        dim
):
    engine = neko_slice;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_integrated_tensor: integrated_tensor, engine.OUTPUT_names: names},
        engine.PARAM_dim: dim, "modcvt_dict": {}}}
# let's not starving our experts
class neko_slice_based(neko_slice):
    PARAM_base="base";
    def set_etc(this, params):
        super().set_etc(params);
        this.base=neko_get_arg(this.PARAM_base,params,0.3);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        integrated_tensor = workspace.get(this.integrated_tensor);
        v=this.base/integrated_tensor.shape[this.dim];
        based=integrated_tensor*(1-this.base)+v;
        for i in range(len(this.names)):
            workspace.add(this.names[i],torch.select(based,this.dim,i))
        return workspace, environment;

def get_neko_slice_based(
        integrated_tensor,
        names,
        dim
):
    engine = neko_slice_based;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_integrated_tensor: integrated_tensor, engine.OUTPUT_names: names},
        engine.PARAM_dim: dim, "modcvt_dict": {}}}


# divide evenly.
class neko_fake_slice_even(neko_slice):
    PARAM_base="base";
    def set_etc(this, params):
        super().set_etc(params);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        integrated_tensor = workspace.get(this.integrated_tensor);
        v=1/integrated_tensor.shape[this.dim];
        for i in range(len(this.names)):
            workspace.add(this.names[i],torch.select(v,this.dim,i))
        return workspace, environment;


def get_neko_fake_slice_even(
        integrated_tensor,
        names,
        dim
):
    engine = neko_fake_slice_even;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_integrated_tensor: integrated_tensor, engine.OUTPUT_names: names},
        engine.PARAM_dim: dim, "modcvt_dict": {}}}
# let's not starving our experts

if __name__ == '__main__':
    neko_slice.print_default_setup_scripts();

