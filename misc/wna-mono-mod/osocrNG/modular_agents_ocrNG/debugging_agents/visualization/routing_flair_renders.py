import cv2

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

import numpy as np
class neko_routing_flair_making_agent(neko_module_wrapping_agent):
    INPUT_router_actions = "router_actions";
    OUTPUT_flairs= "routing_flair";
    PARAM_node_colors="node_colors";
    def  set_mod_io(this,iocvt_dict,modcvt_dict):
        this.router_actions = this.register_input(this.INPUT_router_actions, iocvt_dict);
        this.flairs = this.register_output(this.OUTPUT_flairs, iocvt_dict);
        pass;
    def  set_etc(this,params):
        this.node_colors = neko_get_arg(this.PARAM_node_colors, params,
                                        ["0aefff","580aff","be0aff","ff8700","ff0000","ffd300","a1ff0a","deff0a","0aff99","147df5"]);
        pass;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        router_actions = workspace.get(this.router_actions);
        fls=[];
        for l in router_actions:
            aaf=[];
            for a in l:
                rgbf=np.zeros([32, 32, 3], dtype=np.uint8)+np.array([int(this.node_colors[a][i:i+2], 16) for i in (0, 2, 4)],dtype=np.uint8);
                aaf.append(cv2.cvtColor(rgbf,cv2.COLOR_RGB2BGR));
            fls.append([np.concatenate(aaf)]);
        workspace.add(this.flairs,fls)
        return workspace,environment;
def get_neko_routing_flair_making_agent(
    router_actions,
    flairs,
    node_colors
):
    engine = neko_routing_flair_making_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_router_actions: router_actions, engine.OUTPUT_flairs: flairs}, engine.PARAM_node_colors: node_colors, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_routing_flair_making_agent.print_default_setup_scripts();
