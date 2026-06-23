import multiprocessing

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from typing import Dict
import inspect
class neko_abstract_agent:
    # maps a local name to the global name, and returns the global name.
    SETUPCFGS={
        "PARAM": ["NEP_root_NEP","params"],
        "INPUT": ["\"iocvt_dict\"","inputs"],
        "OUTPUT": ["\"iocvt_dict\"","outputs"],
        "MOD": ["\"modcvt_dict\"","mods"],

    }
    PORDER=["inputs","outputs","mods","params"];
    @classmethod
    def config_term(cls,term:str,cfgd:Dict,pdict:Dict,dftdict:Dict):
        if(term.find("_") <0):
            return cfgd,pdict,dftdict;
        f,c=term.split("_",1);
        if(f=="DFT"):
            dftdict[c]=c;
        if(f not in cls.SETUPCFGS):
            return cfgd,pdict,dftdict;
        cfgl,pl=cls.SETUPCFGS[f];
        if(cfgl == "NEP_root_NEP"):
            cfgd["engine."+term]=c;
            pdict[pl].append(c);
        else:
            if(cfgl not in cfgd):
                cfgd[cfgl]={};
            pdict[pl].append(c);
            cfgd[cfgl]["engine."+term]=c;
        return cfgd, pdict, dftdict;

    @classmethod
    def get_default_configuration_dict(cls):

        aks=inspect.getmembers(cls);
        cfgd={};
        dftdict={};
        pdict={
            "inputs":[],
            "mods":[],
            "outputs":[],
            "params":[],
        }
        for k in aks:
            cfgd,pdict,dftdict=cls.config_term(k[0],cfgd,pdict,dftdict);
        cfgd={"\"agent\"": 'engine', "\"params\"":cfgd};
        if("\"modcvt_dict\"" not in cfgd["\"params\""]):
            cfgd["\"params\""]["\"modcvt_dict\""]={};
        return cfgd,pdict,dftdict;
    @classmethod
    def get_default_configuration_scripts(cls):
        cfgd, pdict,dftdict=cls.get_default_configuration_dict();
        pstr=["@classmethod","def get_agtcfg(cls,"];
        for section in cls.PORDER:
            if (len(pdict[section])):
                pstr += ["    " + ",".join(pdict[section]) + ","];
        pstr[-1] = pstr[-1][:-1];
        pstr += ["):"];
        pstr+=["    return "+str(cfgd).replace("\'engine.","cls.").replace("\'","").replace("engine,","cls,")];
        pstr+=["def get_"+cls.__name__+"("];
        for section in cls.PORDER:
            if(len(pdict[section])):
                pstr+=["    "+",".join(pdict[section])+","];
        pstr[-1]=pstr[-1][:-1];
        pstr+=["):"];
        pstr+=["    engine = "+cls.__name__+";\n"+"    return "+str(cfgd).replace("\'engine.","engine.").replace("\'","")];
        return pstr;
    @classmethod
    def print_default_configuration_scripts(cls):
        for i in cls.get_default_configuration_scripts():
            print(i);
    def register(this,local_name,param,cvtdict,default=None):
        global_name=neko_get_arg(local_name,param,default);
        cvtdict[local_name]=global_name;
        return global_name;

    def register_list(this, local_name, param,cvtdict, default=None):
        nms = this.register(local_name, param, {}, default);
        for i in range(len(nms)):
            cvtdict[local_name + "_"+str(i)]=nms[i];
        return nms;
    def setup(this,param):
        pass;



    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        print("Not implemented", workspace.epoch_idx,",",workspace.batch_idx);
        workspace.batch_idx=workspace.batch_idx+1;
        return workspace,environment;

    def __init__(this,param):
        this.setup(param);


class neko_abstract_sync_agent(neko_abstract_agent):
    pass;


# Note that there are no rules preventing you to create agents inside agents.
# We updated this class so there are less running wild characters.
class neko_abstract_async_agent(neko_abstract_agent):
    STATUS_running="running";
    STATUS_stopped="stopped";

    PARAM_remapping="remapping";
    PARAM_2_remapping_queues="queues";
    PARAM_2_remapping_assets="assets";

    PARAM_INPUT_QUEUES="inputs";
    PARAM_OUTPUT_QUEUES="outputs";
    # sets
    def grab_nested(this, term, object_dict):
        if (type(term) is list):
            return [this.grab_nested(n, object_dict) for n in term];
        else:
            return object_dict[term];
    def remap(this,remapping_dict,source_dict):
        result={};
        for k in remapping_dict:
            result[k]=this.grab_nested(remapping_dict[k],source_dict);
        return result;

    def make_private_workspace(this, workspace, environment, cvt_dict):
        pworkspace = neko_workspace();
        if(workspace is not None):
            pworkspace.batch_idx=workspace.batch_idx;
            pworkspace.epoch_idx = workspace.epoch_idx;
        for ik in cvt_dict:
            pworkspace.inter_dict[ik] = environment.queue_dict[ik].get();
        return pworkspace;
    def make_private_enviroment(this,environment,remapdict):
        penvironment = neko_environment();
        penvironment.queue_dict = this.remap(
            remapdict[this.PARAM_2_remapping_queues], environment.queue_dict);
        penvironment.assets_dict = this.remap(
            remapdict[this.PARAM_2_remapping_assets], environment.assets_dict);
        return penvironment
    def feedback(this,privateworkspace,private_workspace,workspace,environment):

        pass;

    def take_action_private(this, workspace, environment):
        pworkspace=this.make_private_workspace(workspace,environment,this.inputs)
        pworkspace,penv = this.take_action(pworkspace, environment);
        this.feedback(pworkspace,penv,workspace, environment);
        return workspace,environment;

        # if you do need sharing memory between
        # Pipes, memory, scheduling, latching and locking....
        # It sounds like agents are process----yes you are CORRECT!
        # The dead knowledge comes back at us! ZOMBIE attack!
        # del workspace;
        pass;

    def setup(this,param):
        this.input_dict = {};
        this.output_dict = {};
        this.epoch_cnt=9;
        this.iter_cnt=39;
        pass;


    def action_loop(this):
        # workspace CAN be semi-permenent, however without promise.
        # please assume that they are NOT.
        # Eventually, the user need to do the bookkeeping.
        print("loop undefined");
        workspace=neko_workspace();
        for i in range(this.epoch_cnt):
            for j in range(this.iter_cnt):
                this.take_action_private(workspace,this.environment);
                workspace.batch_idx+=1;
            workspace.epoch_idx+=1;
            workspace.batch_idx=0;

    def mount_environment(this,param,environment):
        this.inputs=[];
        this.outputs=[];
        if(this.PARAM_INPUT_QUEUES in param):
            this.inputs=param[this.PARAM_INPUT_QUEUES];
        if(this.PARAM_OUTPUT_QUEUES in param):
            this.outputs=param[this.PARAM_OUTPUT_QUEUES];
        if(this.PARAM_remapping not in param):
            this.environment = environment;
        else:
            this.environment=this.make_private_enviroment(environment,param[this.PARAM_remapping])
    def __init__(this,param):
        super().__init__(param);
        this.worker=None;
        this.stop();
        pass;

    def start(this,mapping_param,environment,mode="forkserver"):
        this.stop();
        this.mount_environment(mapping_param,environment);
        this.status=this.STATUS_running;
        if(mode=="forkserver"):
            this.worker=multiprocessing.get_context(mode).Process(target=this.action_loop);
        else:
            this.worker = multiprocessing.get_context(mode).Process(target=this.action_loop);

        this.worker.start();
        pass;
    def start_sync(this,mapping_param,environment,mode="forkserver"):
        this.stop();
        this.mount_environment(mapping_param, environment);
        this.status = this.STATUS_running;
        this.action_loop();

    def wait(this):
        this.worker.join();
    def stop(this):
        this.input_dict = {};
        this.output_dict = {};
        this.module_dict={};
        if(this.worker is not None):
            this.worker.kill();
        # Well the worker can have children, so there is no need to make it a list here.
        this.worker=None;
        this.status=this.STATUS_stopped;
        this.environment=neko_environment();
        this.worker = None;
        this.workspace = None;
    def stop_and_quit(this):
        this.stop();
        exit(0);

if __name__ == '__main__':
    a=neko_abstract_async_agent(
        {}
    )
    a.start({"inputs":[],
         "outputs":[]},neko_environment());
    a.wait();
