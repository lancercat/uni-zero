# assumes we have one single agent for single input
# the agent will pick RANDOMLY if there are more than one agents assigned to the same input.
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment



# upon exhaustive routing, the id is just 0 to N
class default_id_agent(neko_module_wrapping_agent):
    INPUT_ref="ref";
    OUTPUT_IDs="ids";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.ref = this.register_input(this.INPUT_ref, iocvt_dict);
        this.IDs = this.register_output(this.OUTPUT_IDs, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ref = workspace.get(this.ref);
        ids=list(range(ref.shape[0]));
        workspace.add(this.IDs,ids);
        return workspace, environment;

def get_default_id_agent(
        ref,
        IDs
):
    engine = default_id_agent;
    return {"agent": engine,
            "params": {"iocvt_dict": {engine.INPUT_ref: ref, engine.OUTPUT_IDs: IDs}, "modcvt_dict": {}}}



# this is also slow. for tensors you may want to implement a torch scatter based aggregator.
# But that's beyond the scope of this one.
# we are not anticipating absence or replications yet.
# Both triggers undefined behaviors and can crash your model.
# I need research engineers!!

# this one will just override with the order of the anchors get registered. Don't use unless you have more than one anchor
class neko_simple_aggr_agent(neko_module_wrapping_agent):
    INPUT_SRCs="srcs";
    INPUT_IDs="ids";
    OUTPUT_target="target";
    PARAM_missing_default="missing_default"
    def set_etc(this,param):
        this.misdef=neko_get_arg(this.PARAM_missing_default,param,"NEP_skipped_NEP");
        # this.cnt=neko_get_arg(this.INPUT_TOT_CNT)

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.srcs=this.register_input_list(this.INPUT_SRCs,iocvt_dict);
        this.ids=this.register_input_list(this.INPUT_IDs,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        maxi=-1;
        for i in this.ids:
            if(i not in workspace.inter_dict):
                continue;
            id = workspace.get(i);
            maxi=max(max(id),maxi);
        maxi+=1;
        l=[this.misdef for i in  range(maxi)];
        for s,i in zip(this.srcs,this.ids):
            if(s in workspace.inter_dict and i in workspace.inter_dict):
                id=workspace.get(i);
                dta=workspace.get(s);
                for aid in range(len(id)):
                    bi=id[aid];
                    l[bi]=dta[aid];
        workspace.add(this.aggrtar,l);
        return workspace,environment;

class neko_max_conf_aggr_agent(neko_simple_aggr_agent):
    INPUT_SCRs="in_scrs";


    # this.cnt=neko_get_arg(this.INPUT_TOT_CNT)

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.scrs=this.register_input(this.INPUT_SCRs,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        maxi=-1;
        for i in this.ids:
            if(i not in workspace.inter_dict):
                continue;
            id = workspace.get(i);
            maxi=max(max(id),maxi);
        maxi+=1;
        l=[this.misdef for i in  range(maxi)];
        cscr=[-9999 for i in range(maxi)];
        for s,i,c in zip(this.srcs,this.ids,this.scrs):
            if(s in workspace.inter_dict and i in workspace.inter_dict):
                id=workspace.get(i);
                dta=workspace.get(s);
                scr=workspace.get(c);
                for aid in range(len(id)):
                    bi = id[aid];
                    if(cscr[bi]<scr[aid]):
                        try:
                            l[bi]=dta[aid];
                            cscr[bi] = scr[aid];
                        except:
                            print(bi,len(l),aid,len(dta));
        workspace.add(this.aggrtar,l);
        return workspace,environment;
def get_neko_simple_aggr_agent(IDs_name,SRCs_name,target_name,missing_default,prefixes):
    engine = neko_simple_aggr_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_IDs: IDs_name, engine.INPUT_SRCs: SRCs_name, engine.OUTPUT_target: target_name}, engine.PARAM_missing_default: missing_default, "modcvt_dict": {}}}
def get_neko_max_conf_aggr_agent(
	IDs,SCRs,SRCs,
	target,
	missing_default
):
    engine = neko_max_conf_aggr_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_IDs: IDs, engine.INPUT_SCRs: SCRs, engine.INPUT_SRCs: SRCs, engine.OUTPUT_target: target}, engine.PARAM_missing_default: missing_default, "modcvt_dict": {}}}

if __name__ == '__main__':
    default_id_agent.print_default_setup_scripts();
