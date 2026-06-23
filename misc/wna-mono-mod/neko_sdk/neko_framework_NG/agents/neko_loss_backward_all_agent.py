from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace,neko_environment
# which sums all losses in objdict and commence backward function.
from neko_sdk.log import fatal
class neko_basic_backward_all_agent(neko_abstract_sync_agent):
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(len(workspace.objdict)==0):
            return workspace,environment;
        tl=0;
        for l in workspace.objdict:
            tl=tl+workspace.objdict[l];
            if tl.isnan():
                print(workspace);
                fatal("error!!!!"+l+"IS NAN!!");
        tl.backward();
        # logging is nomore managed by backward agent,
        # we drop the losses that has been back propagated once,
        # so they do not get back propagated for a second time....
        # well if you want more complexity control override this one...
        workspace.objdict={};
        return workspace,environment;

def get_neko_basic_backward_all_agent():
    return {
        "agent":neko_basic_backward_all_agent,
        "params":{}
    }