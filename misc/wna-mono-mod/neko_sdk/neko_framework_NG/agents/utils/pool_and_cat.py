import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace


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
def get_neko_pool_and_cat_agent(
        endpoints,
        features
    ):
        engine = neko_pool_and_cat_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_endpoints: endpoints, engine.OUTPUT_features: features}, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_pool_and_cat_agent.print_default_setup_scripts()
