import torch
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_simple_action_module_wrapping_agent_1i1o_functional
from neko_sdk.cfgtool.argsparse import neko_get_arg
class neko_as_tensor_agent(neko_simple_action_module_wrapping_agent_1i1o_functional):
    PARAM_dtype="dtype";
    PARAM_device="device";
    def op(this,input):
        return torch.tensor(input,device=this.device);

    def set_etc(this, params):
        this.device = neko_get_arg(this.PARAM_device, params);
        this.dtype = neko_get_arg(this.PARAM_dtype, params);
        pass;
    @classmethod
    def get_agtcfg(cls,
                   input,
                   output, dtype,
                   device="cpu"
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_input: input, cls.OUTPUT_output: output}, cls.PARAM_device: device,
                           cls.PARAM_dtype: dtype, "modcvt_dict": {}}}
