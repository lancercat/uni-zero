
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
# just average_stuff_up
class neko_simple_weighted_aggr_agent_nomissing(neko_module_wrapping_agent):
    INPUT_srcs="srcs";
    INPUT_weights="weights"
    INPUT_ids="ids";
    OUTPUT_target="target";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.srcs=this.register_input_list(this.INPUT_srcs,iocvt_dict);
        this.ids=this.register_input_list(this.INPUT_ids,iocvt_dict);
        this.weights=this.register_input_list(this.INPUT_weights,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        t=0;
        for s,lw in zip(this.srcs,this.weights):
            dta=workspace.get(s);
            ws=workspace.get(lw)
            r=dta*ws;
            t=r+t;
        workspace.add(this.aggrtar,t);
        return workspace,environment;
def get_neko_simple_weighted_aggr_agent_nomissing(
    ids,srcs,weights,
    target,
    missing_default
):
    engine = neko_simple_weighted_aggr_agent_nomissing;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_ids: ids, engine.INPUT_srcs: srcs, engine.INPUT_weights: weights, engine.OUTPUT_target: target}, "modcvt_dict": {}}}
# i mean you can
class neko_simple_max_conf_aggr_agent_nomissing_list(neko_module_wrapping_agent):
    INPUT_srcs="srcs";
    INPUT_weights="weights"
    INPUT_ids="ids";
    OUTPUT_target="target";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.srcs=this.register_input_list(this.INPUT_srcs,iocvt_dict);
        this.ids=this.register_input_list(this.INPUT_ids,iocvt_dict);
        this.weights=this.register_input_list(this.INPUT_weights,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        m=list(torch.stack([workspace.get(w) for w in this.weights],-1).argmax(-1).detach().cpu().numpy());
        r=[];
        ds=[workspace.get(s) for s in this.srcs]
        for i in range(len(m)):
            r.append(ds[m[i]][i]);
        workspace.add(this.aggrtar,r);
        return workspace,environment;
def get_neko_simple_max_conf_aggr_agent_nomissing_list(
    ids,srcs,weights,
    target,
    missing_default
):
    engine = neko_simple_max_conf_aggr_agent_nomissing_list;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_ids: ids, engine.INPUT_srcs: srcs, engine.INPUT_weights: weights, engine.OUTPUT_target: target}, "modcvt_dict": {}}}

class neko_simple_weighted_aggr_agent_nomissing_debug_a1(neko_module_wrapping_agent):
    INPUT_srcs="srcs";
    INPUT_weights="weights"
    INPUT_ids="ids";
    OUTPUT_target="target";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.srcs=this.register_input_list(this.INPUT_srcs,iocvt_dict);
        this.ids=this.register_input_list(this.INPUT_ids,iocvt_dict);
        this.weights=this.register_input_list(this.INPUT_weights,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        s=this.srcs[0];
        workspace.add(this.aggrtar,workspace.get(s));
        return workspace,environment;
def get_neko_simple_weighted_aggr_agent_nomissing_debug_a1(
    ids,srcs,weights,
    target,
    missing_default
):
    engine = neko_simple_weighted_aggr_agent_nomissing_debug_a1;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_ids: ids, engine.INPUT_srcs: srcs, engine.INPUT_weights: weights, engine.OUTPUT_target: target}, "modcvt_dict": {}}}



class neko_simple_weighted_aggr_agent(neko_simple_weighted_aggr_agent_nomissing):
    PARAM_missing_default="missing_default"
    def set_etc(this,param):
        this.misdef=neko_get_arg(this.PARAM_missing_default,param,"NEP_skipped_NEP");

        # this.cnt=neko_get_arg(this.INPUT_TOT_CNT)


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        maxi=-1;
        for i in this.ids:
            if(i not in workspace.inter_dict):
                continue;
            id = workspace.get(i);
            maxi=max(max(id),maxi);
        maxi+=1;
        l=[[] for i in  range(maxi)];
        w=[[] for i in range(maxi)];
        r=[None for i in range(maxi)];
        for s,i,lw in zip(this.srcs,this.ids,this.weights):
            if(s in workspace.inter_dict and i in workspace.inter_dict):
                id=workspace.get(i);
                dta=workspace.get(s);
                ws=workspace.get(lw)
                for aid in range(len(id)):
                    bi=id[aid];
                    l[bi].append(dta[aid]);
                    w[bi].append(ws[aid]);

        for i in range(maxi):
            if(len(w[i])):
                al=torch.stack(l[i]);
                slen=len(al.shape);
                wshp=[al.shape[0]]+[1 for _ in range(slen-1)];
                aw=torch.stack(w[i]).view(wshp); # ugly, ikr. pay me a few research engineers and then we talk about elegance.
                r[i]=(aw*al).sum(0)/aw.sum(0);
        workspace.add(this.aggrtar,torch.stack(r));
        return workspace,environment;



def get_neko_simple_weighted_aggr_agent(
    ids,srcs,weights,
    target,
    missing_default
):
    engine = neko_simple_weighted_aggr_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_ids: ids, engine.INPUT_srcs: srcs, engine.INPUT_weights: weights, engine.OUTPUT_target: target}, engine.PARAM_missing_default: missing_default, "modcvt_dict": {}}}


# will deprecate: expotential goes out of this agent.

class neko_simple_weighted_aggr_agentexps(neko_module_wrapping_agent):
    INPUT_srcs="srcs";
    INPUT_weights="weights"
    INPUT_ids="ids";
    OUTPUT_target="target";
    PARAM_missing_default="missing_default"
    def set_etc(this,param):
        this.misdef=neko_get_arg(this.PARAM_missing_default,param,"NEP_skipped_NEP");

        # this.cnt=neko_get_arg(this.INPUT_TOT_CNT)

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.srcs=this.register_input_list(this.INPUT_srcs,iocvt_dict);
        this.ids=this.register_input_list(this.INPUT_ids,iocvt_dict);
        this.weights=this.register_input_list(this.INPUT_weights,iocvt_dict);
        this.aggrtar=this.register_output(this.OUTPUT_target,iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        maxi=-1;
        for i in this.ids:
            if(i not in workspace.inter_dict):
                continue;
            id = workspace.get(i);
            maxi=max(max(id),maxi);
        maxi+=1;
        l=[[] for i in  range(maxi)];
        w=[[] for i in range(maxi)];
        r=[None for i in range(maxi)];
        for s,i,lw in zip(this.srcs,this.ids,this.weights):
            if(s in workspace.inter_dict and i in workspace.inter_dict):
                id=workspace.get(i);
                dta=workspace.get(s);
                ws=workspace.get(lw)
                for aid in range(len(id)):
                    bi=id[aid];
                    l[bi].append(dta[aid]);
                    w[bi].append(ws[aid]);

        for i in range(maxi):
            if(len(w[i])):
                al=torch.stack(l[i]);
                slen=len(al.shape);
                wshp=[al.shape[0]]+[1 for _ in range(slen-1)];
                aw=torch.stack(w[i]).view(wshp); # ugly, ikr. pay me a few research engineers and then we talk about elegance.
                r[i]=(aw*al).sum(0)/aw.sum(0);
        workspace.add(this.aggrtar,torch.stack(r));
        return workspace,environment;
def get_neko_simple_weighted_aggr_agentexps(
    ids,srcs,weights,
    target,
    missing_default
):
    engine = neko_simple_weighted_aggr_agentexps;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_ids: ids, engine.INPUT_srcs: srcs, engine.INPUT_weights: weights, engine.OUTPUT_target: target}, engine.PARAM_missing_default: missing_default, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_simple_weighted_aggr_agent.print_default_setup_scripts()