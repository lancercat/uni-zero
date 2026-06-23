import torch.nn
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf

# We will make parameter free module agents, to reduce coding complexity.
# Collects penalties from all possible decisions.
# Decisions that did not happen has infinite_penalty.

class neko_penalty_collection_agent(neko_module_wrapping_agent):
    INPUT_penalty_name = "penalty";
    INPUT_item_id = "item_id";
    PARAM_anchors="anchors";
    INPUT_router_logits = "router_logits";
    OUTPUT_penaltyspace = "penaltyspace";
    OUTPUT_penaltymask = "penaltymask";

    OUTPUT_weight = "router_logits";
    PARAM_inf_penalty="inf_penalty";
    def set_etc(this,param):
        this.anchors=neko_get_arg(this.PARAM_anchors,param);
        this.inf_penalty=neko_get_arg(this.PARAM_inf_penalty,param,9999);

        for a in this.anchors:
            this.input_dict[a+this.INPUT_item_id]=a+this.item_id;
            this.input_dict[a + this.INPUT_penalty_name] = a + this.penalty;
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.router_logits = this.register_input(this.INPUT_router_logits, iocvt_dict);
        this.item_id=neko_get_arg(this.INPUT_item_id,iocvt_dict);
        this.penalty= neko_get_arg(this.INPUT_penalty_name,iocvt_dict);
        this.penaltyspace = this.register_output(this.OUTPUT_penaltyspace, iocvt_dict);
        this.penaltymask = this.register_output(this.OUTPUT_penaltymask, iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        router_logits=workspace.get(this.router_logits);
        with torch.no_grad():
            penaltyspace=torch.zeros_like(router_logits)+this.inf_penalty;
            penaltymask=torch.zeros_like(router_logits)+1;

            for aid in range(len(this.anchors)):
                if(this.anchors[aid]+this.item_id not in workspace.inter_dict):
                    penaltymask[:, aid][item_id]=0;
                    continue;
                item_id=workspace.get(this.anchors[aid]+this.item_id);
                penalty=workspace.get(this.anchors[aid]+this.penalty);
                penaltyspace[:,aid][item_id]=penalty;

        workspace.add(this.penaltyspace,penaltyspace);
        workspace.add(this.penaltymask,penaltymask);

    @classmethod
    def get_agtcfg(cls,
                   item_id, penalty_name, router_logits,
                   penaltymask, penaltyspace,
                   anchors, inf_penalty
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_item_id: item_id, cls.INPUT_penalty_name: penalty_name,
                           cls.INPUT_router_logits: router_logits, cls.OUTPUT_penaltymask: penaltymask,
                           cls.OUTPUT_penaltyspace: penaltyspace},
            cls.PARAM_anchors: anchors, cls.PARAM_inf_penalty: inf_penalty, "modcvt_dict": {}}}
