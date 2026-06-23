from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent_gen2,neko_abstract_async_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

#
class neko_infinite_hosting_wrapper(neko_abstract_async_agent_gen2):


    PARAM_ACFG="agent_config";
    def setup(this,param):
        super().setup(param);
        this.core_agt=param[this.PARAM_ACFG]["agent"](param[this.PARAM_ACFG]["params"]);

    def mkspace(this,d):
        pass;
    def runstub(this,workspace,environment):
        this.core_agt.take_action(workspace,environment);


#no order preserving. Well we will do oreder preservation later, but not here not now.

# no remapping, just wrap the agent and let it do its thing
class neko_infinite_workspace_shipper(neko_abstract_async_agent_gen2):
    PARAM_INPUT_QUEUES="input";
    PARAM_OUTPUT_QUEUES="output";

    PARAM_in_skipped_prefix="in_skip_prefix";
    PARAM_out_skipped_prefix = "out_skip_prefix";
    PARAM_ACFG="agent_config";
    def setup_loopglobal(this):
        this.coreagt=this.acfg["agent"](this.acfg["params"]); # if one of the subagent ever creates a mp.Pool, this gonna dead lock.

    def setup(this, param):
        super().setup(param);
        this.acfg = param[this.PARAM_ACFG];

        this.iskp = neko_get_arg(this.PARAM_in_skipped_prefix, param, []);
        this.oskp = neko_get_arg(this.PARAM_out_skipped_prefix, param, []);


    def mkspace(this, d):
        pass;

    def runstub(this,workspace,environment):
        # print("processing",workspace.inter_dict)
        this.coreagt.take_action(workspace,environment);
        # print("processed")


    @classmethod
    def get_agtcfg(cls,
                   input_queue,inames,
                   output_queue,onames,
                   acfg
                   ):
        if(input_queue is not None):
            iqs=[input_queue];
        else:
            iqs=[];
        if (output_queue is not None):
            oqs = [output_queue];
        else:
            oqs = [];

        return {"agent": cls, "params": {cls.PARAM_INPUT_QUEUES:iqs,cls.PARAM_INAMES: [inames],
                           cls.PARAM_OUTPUT_QUEUES:oqs,cls.PARAM_ONAMES: [onames], cls.PARAM_ACFG: acfg}}
# no order preserving. Well we will do oreder preservation later, but not here not now.


