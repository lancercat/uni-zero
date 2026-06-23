from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class neko_detach_agent(neko_module_wrapping_agent):
    INPUT_input="input";
    OUTPUT_output = "output";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_input, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        input = workspace.get(this.input);
        workspace.add(this.output,input.detach())
        return workspace, environment;

def get_neko_detach_agent(
        input,
        output
):
    engine = neko_detach_agent;
    return {"agent": engine,
            "params": {"iocvt_dict": {engine.INPUT_input: input, engine.OUTPUT_output: output}, "modcvt_dict": {}}
            }

class neko_detach_list_agent(neko_module_wrapping_agent):
    INPUT_input="input";
    OUTPUT_output = "output";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_input, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        os = [workspace.get(i).detach() for i in this.input] ;
        for k,v in zip(this.output,os):
            workspace.add(k,v);
        return workspace, environment;

def get_neko_detach_list_agent(
        input,
        output
):
    engine = neko_detach_list_agent;
    return {"agent": engine,
            "params": {"iocvt_dict": {engine.INPUT_input: input, engine.OUTPUT_output: output}, "modcvt_dict": {}}}
