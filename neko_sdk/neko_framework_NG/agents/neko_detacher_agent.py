from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace,neko_environment
# which sums all losses in objdict and commence backward function.
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent

# create detaches copy of tensors.
# will be used to warp stuff between processes.
class neko_detacher_agent(neko_module_wrapping_agent):
    INPUT_to_detach="to_detach";
    OUTPUT_detached="detached";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.in_name=this.register_input(this.INPUT_to_detach,iocvt_dict);
        this.out_name=this.register_output(this.OUTPUT_detached, iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.out_name, workspace.get(this.in_name).detach());
        return workspace,environment
class neko_list_detacher_agent(neko_detacher_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.in_names=this.register_input_list(this.INPUT_to_detach,iocvt_dict);
        this.out_names=this.register_output_list(this.OUTPUT_detached, iocvt_dict);
        assert len(this.in_names)==len(this.out_names);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for i,o in zip(this.in_names,this.out_names):
            workspace.add(o,[t.detach()for t in workspace.get(i)]);
        return workspace,environment

def get_neko_detacher_agent(to_detach_name,
detached_name
):
    engine = neko_detacher_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_to_detach: to_detach_name, engine.OUTPUT_detached: detached_name}, "modcvt_dict": {}}}
def get_neko_list_detacher_agent(to_detach_name,
detached_name
):
    engine = neko_list_detacher_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_to_detach: to_detach_name, engine.OUTPUT_detached: detached_name}, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_detacher_agent.get_default_configuration_scripts());