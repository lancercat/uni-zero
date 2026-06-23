import torch

from neko_sdk.log import fatal
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_cat_agent(neko_module_wrapping_agent):
    INPUT_input_names="input_names";
    OUTPUT_output = "output";
    PARAM_catdim="catdim";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.inputs = this.register_input_list(this.INPUT_input_names, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.catdim=neko_get_arg(this.PARAM_catdim,params);

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        inputs = [workspace.get(i) for i in this.inputs];
        workspace.add(this.output,torch.cat(inputs,dim=this.catdim));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   inputs,
                   output,
                   catdim
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_input_names: inputs, cls.OUTPUT_output: output}, "modcvt_dict": {},cls.PARAM_catdim:catdim}}

class neko_singlelist_cat_agent(neko_module_wrapping_agent):
    INPUT_input="input";
    OUTPUT_output = "output";
    PARAM_catdim="catdim";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.inputs = this.register_input(this.INPUT_input, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.catdim=neko_get_arg(this.PARAM_catdim,params);

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        inputs= workspace.get(this.inputs);
        if len(inputs)==0:
            fatal("empty batch");
        workspace.add(this.output,torch.cat(inputs,dim=this.catdim));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   inputs,
                   output,
                   catdim
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_input: inputs, cls.OUTPUT_output: output}, "modcvt_dict": {},cls.PARAM_catdim:catdim}}


# used to extract global representation from FPN endpoints
class neko_pool_and_cat_agent(neko_module_wrapping_agent):
    INPUT_endpoints="endpoints";
    OUTPUT_features = "features";
    def  set_mod_io(this,iocvt_dict,modcvt_dict):
        this.endpoints = this.register_input(this.INPUT_endpoints, iocvt_dict);
        this.features =  this.register_output(this.OUTPUT_features, iocvt_dict);
        pass;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        endpoints = workspace.get(this.endpoints);
        ofe=torch.cat([e.mean(-1).mean(-1) for e in endpoints],dim=-1);
        workspace.add(this.features,ofe);
        return workspace,environment;
    @classmethod
    def get_agtcfg(cls,
                   endpoints,
                   features
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_endpoints: endpoints, cls.OUTPUT_features: features},
                                         "modcvt_dict": {}}}
def get_neko_pool_and_cat_agent(
        endpoints,
        features
    ):
        engine = neko_pool_and_cat_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_endpoints: endpoints, engine.OUTPUT_features: features}, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_pool_and_cat_agent.print_default_setup_scripts()
