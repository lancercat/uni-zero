
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
# Filters applied pre_routing, can be chained
from osocrNG.modular_agents_ocrNG.filter_agents.abstract_pre_filter_agent import neko_pre_filter_agent

# drops samples that do not fit an expert for obvious reason
# restrict_mask defines what information would be "restricted" to what experts.
class neko_pre_length_filter_agent(neko_pre_filter_agent):
    INPUT_raw_label="raw_label";
    PARAM_max_lens="maxlens";
    PARAM_min_lens="minlens";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.label=this.register_input(this.INPUT_raw_label,iocvt_dict);
    def set_etc(this,param):
        this.maxlens=neko_get_arg(this.PARAM_max_lens,param);
        this.acnt=len(this.maxlens);
        this.minlens=neko_get_arg(this.PARAM_min_lens,param,"NEP_skipped_NEP");
        if(this.minlens is None):
            this.minlens=[-9 for _ in this.maxlens]

    def filter(this,raw_ims,basemsk,workspace:neko_workspace,environment:neko_environment):
        ll=[len(l) for l in workspace.get(this.label)];
        for i in range(basemsk.shape[0]):
            for t in range(len(this.maxlens)):
                if (ll[i]<this.minlens[t] or ll[i]>this.maxlens[t]):
                    basemsk[i][t]=0;
        return basemsk;
def get_neko_pre_length_filter_agent(label_name,raw_ims_name,restrict_mask_in_name,
restrict_mask_out_name,max_lens,min_lens):
    engine = neko_pre_length_filter_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_raw_label: label_name, engine.INPUT_raw_ims: raw_ims_name, engine.INPUT_restrict_mask_in: restrict_mask_in_name, engine.OUTPUT_restrict_mask_out: restrict_mask_out_name}, engine.PARAM_max_lens: max_lens, engine.PARAM_min_lens: min_lens, "modcvt_dict": {}}}
