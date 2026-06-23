from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace,neko_environment
# which sums all losses in objdict and commence backward function.

class neko_optim_agent(neko_abstract_sync_agent):
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        environment.modset.norm_grad();
        environment.modset.update_para();
        environment.modset.zero_grad();


def get_neko_optim_agent():
    return {
        "agent":neko_optim_agent,
        "params":{}
    }