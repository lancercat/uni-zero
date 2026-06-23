import torch
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.CnC.command.agents.policy_sampling.abstract_sampling import neko_abstract_policy_agent


class neko_bestk_policy_agent(neko_abstract_policy_agent):
    PARAM_k="k";
    def set_etc(this,param):
        super().set_etc(param);
        this.k=neko_get_arg(this.PARAM_k,param,9999); # well this is exhaustive

    def sample(this, logits, mask, log_probs):
        actions = [[] for _ in logits];
        with torch.no_grad():
            mlogits = logits.cpu() + ( mask-1) * 999999999;
            actions_ = torch.argsort(mlogits, dim=-1, descending=True).cpu().numpy();
            dnm=mask.cpu().numpy();
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    a=actions_[i][j];
                    if (dnm[i][a] <= 0.5): # float issues duh
                        continue;
                    if(len(actions[i])==this.k):
                        break;
                    actions[i].append(a);
        return actions;
def get_neko_bestk_policy_agent(allow_mask_name,states_name,
actions_name,logits_name,logp_name,
router_name,
k):
    engine = neko_bestk_policy_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_allow_mask: allow_mask_name, engine.INPUT_states: states_name, engine.OUTPUT_actions: actions_name, engine.OUTPUT_logits: logits_name, engine.OUTPUT_logp: logp_name}, "modcvt_dict": {engine.MOD_router: router_name}, engine.PARAM_k: k}}

if __name__ == '__main__':
    print(neko_bestk_policy_agent.get_default_configuration_scripts());