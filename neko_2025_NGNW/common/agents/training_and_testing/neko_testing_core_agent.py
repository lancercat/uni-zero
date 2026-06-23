from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_2025_NGNW.common.agents.training_and_testing.neko_32x_meta_caching_agent import neko_320_im_meta_caching_agent
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_sdk.neko_framework_NG.names import P,PLST


class neko_task_testing_core_agent(neko_module_wrapping_agent):
    PARMA_testing_net_cfg="testing_net_cfg";
    PARAM_testing_data_cfg="testing_data_cfg";
    PARAM_testing_logging_cfg="testing_logging_cfg";
    PARAM_name="name"
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        pass;

    def set_etc(this, params):
        this.testing_data_cfg = neko_get_arg(this.PARAM_testing_data_cfg, params);
        this.recorder_agts:awa=awa.make(neko_get_arg(this.PARAM_testing_logging_cfg,params));
        this.net_agt:awa=awa.make(neko_get_arg(this.PARMA_testing_net_cfg,params));
        this.data_agt:awa=awa.make(this.testing_data_cfg);
        this.da = this.data_agt.get_agt_at(0);
        this.name=neko_get_arg(this.PARAM_name,params,"default_string");

        this.numbatch=this.da.len();

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for an in this.recorder_agts.agent_d:
            this.recorder_agts.agent_d[an].reset();
        this.da.reset();
        for i in range(this.numbatch):
            wsi=workspace.shallow_fork(); # the meta is for all batches
            wsi.add("benchmark_name",this.name,True);
            this.data_agt.take_action(wsi, environment);
            this.net_agt.take_action(wsi,environment);
            for an in this.recorder_agts.agent_d:
                this.recorder_agts.agent_d[an].take_action(wsi,environment);
            pass;
        for an in this.recorder_agts.agent_d:
            this.recorder_agts.agent_d[an].report(workspace, environment);
            pass;
        pass;
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,testing_data_cfg,testing_net_cfg,test_logging_cfg,name="default_string"):
        return {"agent": cls, "params": {
                                        cls.PARMA_testing_net_cfg:testing_net_cfg,
                                         cls.PARAM_testing_data_cfg: testing_data_cfg,
                                        cls.PARAM_testing_logging_cfg: test_logging_cfg,
                                        cls.PARAM_name:name,
            "modcvt_dict": {},"iocvt_dict":{}}}



# let's see what we need before we stage this as generic.
# do we do one data many meta? maybe but in another agent...

class neko_task_testing_agent(neko_module_wrapping_agent):
    PARMA_testing_net_cfg="testing_net_cfg";
    PARAM_testing_data_cfg="testing_data_cfg";
    PARAM_testing_meta_cfg="testing_meta_cfg";
    PARAM_testing_logging_cfg="testing_logging_cfg";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        pass;

    def set_etc(this, params):

        this.testing_data_cfg = neko_get_arg(this.PARAM_testing_data_cfg, params);
        this.testing_meta_cfg = neko_get_arg(this.PARAM_testing_meta_cfg, params);
        this.recorder_agts:awa=awa.make(neko_get_arg(this.PARAM_testing_logging_cfg,params));
        this.net_agt:awa=awa.make(neko_get_arg(this.PARMA_testing_net_cfg,params));
        this.meta_agt:neko_320_im_meta_caching_agent=neko_320_im_meta_caching_agent.make(this.testing_meta_cfg);
        this.data_agt:awa=awa.make(this.testing_data_cfg);
        da = this.data_agt.get_agt_at(0);
        this.numbatch=da.len();

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ws = neko_workspace();
        this.meta_agt.take_action(ws,environment);
        for an in this.recorder_agts.agent_d:
            this.recorder_agts.agent_d[an].reset();

        for i in range(this.numbatch):
            wsi=ws.shallow_fork(); # the meta is for all batches
            this.data_agt.take_action(wsi, environment);
            this.net_agt.take_action(wsi,environment);
            for an in this.recorder_agts.agent_d:
                this.recorder_agts.agent_d[an].take_action(wsi,environment);
            pass;
        for an in this.recorder_agts.agent_d:
            this.recorder_agts.agent_d[an].report(ws, environment);
            pass;
        pass;
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,testing_net_cfg,testing_data_cfg, testing_meta_cfg,test_logging_cfg):
        return {"agent": cls, "params": {
                                        cls.PARMA_testing_net_cfg:testing_net_cfg,
                                         cls.PARAM_testing_data_cfg: testing_data_cfg,
                                         cls.PARAM_testing_meta_cfg: testing_meta_cfg,
                                        cls.PARAM_testing_logging_cfg: test_logging_cfg,
            "modcvt_dict": {},"iocvt_dict":{}}}

