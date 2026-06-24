from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.names import default_ocr_variable_names as dvn
from neko_sdk.cfgtool.argsparse import neko_get_arg

# drops samples that do not fit an expert for obvious reason
class post_neko_length_filter_agent(neko_module_wrapping_agent):
    OUTPUT_weightayloads="outpayloads";
    INPUT_payloads="inpayloads";
    INPUT_len="len";
    PARAM_max_len="maxlen";
    PARAM_min_len="minlen";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.inp=this.register_input_list(this.INPUT_payloads,iocvt_dict);
        this.oup=this.register_output_list(this.OUTPUT_weightayloads,iocvt_dict);
        this.len=this.register_input(this.INPUT_len,iocvt_dict);
    def set_etc(this,param):
        this.maxlen=neko_get_arg(this.PARAM_max_len,param);
        this.minlen=neko_get_arg(this.PARAM_min_len,param,-9); # unless you want some forced load balance
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        ll=workspace.get(this.len);
        for i in range(len(ll)):
            if (ll[i]<this.minlen or ll[i]>this.maxlen):
                continue;
            for t in range(len(this.inp)):
                workspace.append_add(
                    workspace.get(this.oup[t]),
                    workspace.get(this.inp[t])[i]
                );
class post_neko_length_filter_agent(neko_module_wrapping_agent):
    OUTPUT_weightayloads="outpayloads";
    INPUT_payloads="inpayloads";
    INPUT_len="len";
    PARAM_max_len="maxlen";
    PARAM_min_len="minlen";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.inp=this.register_input_list(this.INPUT_payloads,iocvt_dict);
        this.oup=this.register_output_list(this.OUTPUT_weightayloads,iocvt_dict);
        this.len=this.register_input(this.INPUT_len,iocvt_dict);
    def set_etc(this,param):
        this.maxlen=neko_get_arg(this.PARAM_max_len,param);
        this.minlen=neko_get_arg(this.PARAM_min_len,param,-9); # unless you want some forced load balance
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        ll=workspace.get(this.len);
        for i in range(len(ll)):
            if (ll[i]<this.minlen or ll[i]>this.maxlen):
                continue;
            for t in range(len(this.inp)):
                workspace.append_add(
                    workspace.get(this.oup[t]),
                    workspace.get(this.inp[t])[i]
                );



if __name__ == '__main__':
    print(neko_length_filter_agent.get_default_configuration_dict())
