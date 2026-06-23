from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

# we assume all you need DURING TEST are the centers and centers ONLY.
# we do not do ducky shit during testing, duh.
# also, we do not load sideinfo for nothing during inference, either..

# the input agent manage the meta path--- i know its ugly but...
# kiss. if we keep the path, we make it static.
# agents are made to be replaced anytime anyway.
# one modality.
class neko_key_repo_one(ama):
    PARAM_proto_agent_para="proto_agent_param";

    OUTPUT_proto="proto";
    OUTPUT_tdict="tdict";
    OUTPUT_proto_utf = "proto_utf";
    # plabel? Oh yes, plabel is now managed by heads, don't mess with that.


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto = this.register_output(this.OUTPUT_proto, iocvt_dict);
        this.proto_utf = this.register_output(this.OUTPUT_proto_utf, iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_tdict, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.capacity = neko_get_arg(this.PARAM_capacity, params);
        agt_param=neko_get_arg(this.PARAM_proto_agent_para, params);
        this.proto_agent=agt_param["agent"](agt_param["params"]);
        this.cache=None;
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.cache is None):
            ws=neko_workspace();
            this.proto_agent(ws,environment);
            this.cache=(
                ws.get(this.proto),
                ws.get(this.tdict),
                ws.get(this.proto_utf),
            );
        else:
            workspace.add(this.proto,this.cache[0]);
            workspace.add(this.tdict,this.cache[1]);
            workspace.add(this.proto_utf,this.cache[2]);

        return workspace, environment;
