from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
import torch
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
# for a list of same sized tensors we can do this....
class neko_concated_rgb_mvn_agent(ama):
    PARAM_device="device";
    PARAM_dtype="dtype";
    PARAM_mean="mean";
    PARAM_var="var";

    INPUT_raw_samples="raw_samples";
    OUTPUT_normed_tensor_samples="tensor_samples";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.raw_samples = this.register_input(this.INPUT_raw_samples, iocvt_dict);
        this.normed_tensor_samples = this.register_output(this.OUTPUT_normed_tensor_samples, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.device = neko_get_arg(this.PARAM_device, params,"cpu");
        this.dtype = neko_get_arg(this.PARAM_dtype, params,torch.float);
        mean = neko_get_arg(this.PARAM_mean, params);
        var = neko_get_arg(this.PARAM_var, params);
        this.mean = torch.nn.Parameter(torch.tensor(mean,dtype=this.dtype,device=this.device).squeeze(0), False);
        this.var = torch.nn.Parameter(torch.tensor(var,dtype=this.dtype,device=this.device).squeeze(0), False);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_samples = workspace.get(this.raw_samples);
        imgl = [(torch.tensor(i, dtype=this.dtype).permute(2, 0, 1).unsqueeze(0).to(
            this.device) - this.mean) / this.var for i in raw_samples];
        workspace.add(this.normed_tensor_samples,imgl);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   raw_samples,
                   tensor_samples,
                   mean=None, var=None,
                   device=torch.device("cpu"),
                   dtype=torch.dtype(torch.float32)
                   ):
        if(mean is None):
            mean=[127.5];
        if (var is None):
            mean = [128];
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_raw_samples: raw_samples, cls.OUTPUT_tensor_samples: tensor_samples},
            cls.PARAM_device: device, cls.PARAM_dtype: dtype, cls.PARAM_mean: mean, cls.PARAM_var: var,
            "modcvt_dict": {}}}




if __name__ == '__main__':
    neko_list_rgb_mvn_agent.print_default_setup_scripts();