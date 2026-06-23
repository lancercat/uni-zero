from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# just assign. may be we will later introduce a more complex copying based one.
# why don't we just change how things are wired up? we can do that, till we start an multi-gpu layout.
# to have a unified api, we have this third wheel.
class neko_broadcasting_agent_static_single_dev_just_assign(neko_module_wrapping_agent):
    PARAM_prefix = "prefix";
    PARAM_routing_subjects= "routing_subjects";
    PARAM_targets="targets";


    def set_etc(this, param):
        # this.sample_number=neko_get_arg(this.PARAM_sample_number,param,3);
        this.routing_subjects = neko_get_arg(this.PARAM_routing_subjects, param);
        this.targets=neko_get_arg(this.PARAM_targets,param);
        this.prefix = neko_get_arg(this.PARAM_prefix, param, "");
        for ni in this.routing_subjects:
            this.input_dict[ni]=ni;
            for pfx in this.targets:
                tarname = this.prefix+pfx + ni.replace(this.prefix, "");
                this.output_dict[tarname]=tarname;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for ni in this.routing_subjects:
            if(ni in workspace.inter_dict):
                t=workspace.get(ni); # if something is missing, we skip them
                for pfx in this.targets:
                    tarname = this.prefix+ pfx + ni.replace(this.prefix, "");
                    workspace.add(tarname,t);
        return workspace,environment;
def get_neko_broadcasting_agent_static_single_dev_just_assign(prefix,routing_subjects,targets):
    engine = neko_broadcasting_agent_static_single_dev_just_assign;
    return {"agent": engine,
            "params": {
                engine.PARAM_prefix: prefix,
                engine.PARAM_routing_subjects: routing_subjects,
                engine.PARAM_targets: targets,
                "modcvt_dict": {},
                "iocvt_dict": {},
            }
            }

if __name__ == '__main__':
    print(neko_broadcasting_agent_static_single_dev_just_assign.get_default_configuration_scripts())