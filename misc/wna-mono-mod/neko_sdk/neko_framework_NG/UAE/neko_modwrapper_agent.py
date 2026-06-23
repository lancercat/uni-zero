# NG hoards a bunch of agents(mod). within a bunch of agents.
import copy
import time

import torch.cuda
from easydict import EasyDict

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch import multiprocessing as trmp
from torch import distributed as trd
#Agents are not usually meant for data processing, data processing is controlled in the bogomods
#agents are here to decide which module to call and to produce what in the dictionary. Thus, it is usally nothing but a wrapper.
from functools import partial
from torch.nn.parallel import parallel_apply
# in simple language, do not process tensors directly here. use modules and bogos to manipulate them.

class neko_module_wrapping_agent(neko_abstract_sync_agent):
    SETUP_MOD_IO={
        "inputs": ["this.register_input","INPUT_","iocvt_dict"],
        "outputs": ["this.register_output","OUTPUT_","iocvt_dict"],
        "mods": ["this.register_mod","MOD_","modcvt_dict"],
    }
    SETUP_ETC={
        "params": ["neko_get_arg","PARAM_","params"],
    }
    ACTION=["inputs"];
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        pass;
    def set_etc(this,param):
        pass;
    def setup(this,param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.omods = EasyDict();
        this.mnames = EasyDict();
        this.set_mod_io(param["iocvt_dict"],param["modcvt_dict"]);
        this.set_etc(param);

    def register_input(this,local_name,param,default=None):
        return this.register(local_name,param,this.input_dict,default);
    def register_input_list(this,local_name,param,default=None):
        return this.register_list(local_name,param,this.input_dict,default);

    def register_mod(this, local_name, param, default=None):
        return this.register(local_name, param, this.mnames, default);

    def register_output(this, local_name, param, default=None):
        return this.register(local_name, param, this.output_dict, default);

    def register_output_list(this, local_name, param, default=None):
        return this.register_list(local_name, param, this.output_dict, default);

    @classmethod
    def print_default_setup_scripts(cls):
        for i in cls.get_default_setup_scripts():
            print(i);
        cls.print_default_configuration_scripts();

    @classmethod
    def get_default_setup_scripts(cls):
        cfgd, pdict,dft_dict=cls.get_default_configuration_dict();
        pstr=["def  set_mod_io(this,iocvt_dict,modcvt_dict):"];
        for section in cls.SETUP_MOD_IO:
            for term in pdict[section]:
                if term in dft_dict:
                    pstr.append(
                        "\tthis." + term + " = " + cls.SETUP_MOD_IO[section][0] + "(this." + cls.SETUP_MOD_IO[section][
                            1] + term + ", this." + cls.SETUP_MOD_IO[section][2] + ", " + dft_dict[term] + ");");
                else:
                    pstr.append(
                        "\tthis." + term + " = " + cls.SETUP_MOD_IO[section][0] + "(this." + cls.SETUP_MOD_IO[section][
                            1] + term + ", " + cls.SETUP_MOD_IO[section][2] + ");");
        pstr.append("\tpass;")
        pstr += ["def  set_etc(this,params):"];
        for section in cls.SETUP_ETC:
            for term in pdict[section]:
                if term in dft_dict:
                    pstr.append(
                        "\tthis." + term + " = " + cls.SETUP_ETC[section][0] + "(this." + cls.SETUP_ETC[section][
                            1] + term + ", this." + cls.SETUP_ETC[section][2] + "," + dft_dict[term] + ");");
                else:
                    pstr.append(
                        "\tthis." + term + " = " + cls.SETUP_ETC[section][0] + "(this." + cls.SETUP_ETC[section][
                            1] + term + ", " + cls.SETUP_ETC[section][2] + ");");
        pstr.append("\tpass;")
        pstr+= ["def take_action(this,workspace:neko_workspace,environment:neko_environment):"];
        for section in cls.ACTION:
            for term in pdict[section]:
                pstr.append(
                    "\t"+ term + " = workspace.get(this."+term + ");");
        pstr.append("\treturn workspace,environment;");

        return pstr;

class neko_simple_action_module_wrapping_agent_1i1o(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input="NEP_unconfigured_NEP";
        this.mod = "NEP_unconfigured_NEP";
        this.output = "NEP_unconfigured_NEP";
        fatal("this is a virtual base class, you need to set the input, mod, and output");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.output, environment(this.mod,workspace.get(this.input)));
        return workspace;


class neko_agent_wrapping_agent(neko_abstract_sync_agent):
    PARAM_ACT_VARS="activation_vars";
    PARAM_AGT_LST="agent_list";
    PARAM_disable_till_eid = "disable_till_eid";
    PARAM_disable_till_bid = "disable_till_bid";
    def setup(this,param):
        this.agent_n = [];
        this.agent_d={};
        for name in param[this.PARAM_AGT_LST]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);
        this.activation_vars=neko_get_arg(this.PARAM_ACT_VARS,param,"NEP_skipped_NEP");
        this.disable_till_eid = neko_get_arg(this.PARAM_disable_till_eid, param,0);
        this.disable_till_bid = neko_get_arg(this.PARAM_disable_till_bid, param,"NEP_skipped_NEP");
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(this.disable_till_bid is not None):
            if (workspace.epoch_idx < this.disable_till_eid):
                return workspace,environment;
            elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
                return workspace,environment;

        if(this.activation_vars is not None):
            for v in this.activation_vars:
                if(v not in workspace.inter_dict):
                    return workspace,environment; # if has nothing to do, do nothing.
        for n in this.agent_n:
            # sta=time.time();
            this.agent_d[n].take_action(workspace,environment);
            # end=time.time();
            # print(n,"--takes--",end-sta)
        return workspace,environment;
    @classmethod
    def append_agent_to_cfg(this,cfg,localname,subagt):
        if subagt is None:
            return cfg;
        assert (localname not in cfg["params"]);
        cfg["params"][this.PARAM_AGT_LST].append(localname);
        cfg["params"][localname]=subagt;
        return cfg;

    @classmethod
    def wrap_this(cls, subagt,delay_bid="NEP_skipped_NEP",delay_eid=0):
        return {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["meow"],
                "meow": subagt,
                 cls.PARAM_disable_till_bid: delay_bid,
                 cls.PARAM_disable_till_eid:delay_eid,
            }
        }
    @classmethod
    def empty(cls,actvars=None):
        if(actvars is None):
            actvars=[];
        return  {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [],
                neko_agent_wrapping_agent.PARAM_ACT_VARS: actvars
            }
        }
    @classmethod
    def append_delayed_agent_to_cfg(this,cfg,localname,subagt,delay_bid,delay_eid):
        cfg["params"][this.PARAM_AGT_LST].append(localname);
        cfg["params"][localname]=this.wrap_this(subagt,delay_bid,delay_eid);
        return cfg;
    @classmethod
    def prepend_agent_to_cfg(this, cfg, localname, subagt):
        cfg["params"][this.PARAM_AGT_LST]=[localname]+cfg["params"][this.PARAM_AGT_LST];
        cfg["params"][localname] = subagt;
        return cfg;

# well if you know you are heavy lifting...
class neko_agent_wrapping_agent_nograd(neko_agent_wrapping_agent):
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(this.disable_till_bid is not None):
            if (workspace.epoch_idx < this.disable_till_eid):
                return workspace,environment;
            elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
                return workspace,environment;

        if(this.activation_vars is not None):
            for v in this.activation_vars:
                if(v not in workspace.inter_dict):
                    return workspace,environment; # if has nothing to do, do nothing.
        with torch.no_grad():
            for n in this.agent_n:
                # sta=time.time();
                this.agent_d[n].take_action(workspace,environment);
                # end=time.time();
                # print(n,"--takes--",end-sta)
        return workspace,environment;

# Don't ever do backward with this! it will only sync its stream on the end,
# making backward thread unsafe.
# And it only runs with cuda operations of course
class neko_agent_wrapping_agent_with_cuda_stream(neko_agent_wrapping_agent):
    def setup(this,param):
        super().setup(param);
        this.stream=torch.cuda.Stream();

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        with torch.cuda.stream(this.stream):
            r=super().take_action(workspace,environment);
        this.stream.synchronize();
        return r;
class neko_parallel_agent_wrapping_agent(neko_agent_wrapping_agent):
    @staticmethod
    def execute(environment, workspace_agt):
        workspace, agt = workspace_agt
        agt.take_action(workspace, environment);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if (this.activation_vars is not None):
            for v in this.activation_vars:
                if (v not in workspace.inter_dict):
                    return workspace, environment;  # if has nothing to do, do nothing.
        func = partial(this.execute, environment);
        [None for i in map(func,[(workspace,this.agent_d[n])for n in this.agent_n])]; # wait till every thing ends.q
        return workspace, environment;


class neko_parallel_agent_wrapping_agent_mp(neko_agent_wrapping_agent):
    PARAM_ACT_VARS="activation_vars";
    PARAM_AGT_LST="agent_list";
    PARAM_PARALLEL_CNT="parallel_cnt";
    @staticmethod
    def execute(environment,workspace_agt):
        workspace,agt,warpd=workspace_agt
        agt.take_action(workspace,environment);
        rws=neko_workspace();
        for k in warpd["warp_back"]:
            v=workspace.get(k);
            rws.add(k,copy.copy(v));
        for k in warpd["warp_back_log"]:
            rws.add_log(k,workspace.get_log(k));
        return rws;

    def warp_outbound(this,workspace:neko_workspace,warpd):
        w=neko_workspace();
        for k in warpd["warp_away"]:
            w.add(k,workspace.get(k));
        return w
    def warp_inbound(this,workspace:neko_workspace, sub_workspace:neko_workspace,warpd):
        for k in warpd["warp_back"]:
            workspace.add(k,sub_workspace.get(k));
        for k in warpd["warp_back_log"]:
            workspace.add_log(k,sub_workspace.get_log(k));
    def setup(this,param):
        super().setup(param);
        paracnt=neko_get_arg(this.PARAM_PARALLEL_CNT,param,2);
        this.pool=trmp.get_context("spawn").Pool(processes=paracnt);


        this.agent_n = [];
        this.agent_d={};
        this.agent_m={};
        for name in param[this.PARAM_AGT_LST]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);
            this.agent_m[name]=param["warp_dict"][name];
        this.activation_vars=neko_get_arg(this.PARAM_ACT_VARS,param,"NEP_skipped_NEP");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        warp_env=environment.warp_ref();
        if(this.activation_vars is not None):
            for v in this.activation_vars:
                if(v not in workspace.inter_dict):
                    return workspace,environment; # if has nothing to do, do nothing.
        func = partial(this.execute,warp_env);

        workspaces= this.pool.map(func, [(this.warp_outbound(workspace,this.agent_m[n]) ,this.agent_d[n],this.agent_m[n]) for n in this.agent_n]);
        for n,w in zip(this.agent_n,workspaces):
            this.warp_inbound(workspace,w,this.agent_m[n]);
        return workspace,environment;
    @classmethod
    def append_agent_to_cfg(this,cfg,localname,subagt):
        cfg["params"][this.PARAM_AGT_LST].append(localname);
        cfg["params"][localname]=subagt;
        return cfg;

class neko_parallel_agent_wrapping_agent_ddp(neko_agent_wrapping_agent):
    PARAM_ACT_VARS="activation_vars";
    PARAM_AGT_LST="agent_list";
    PARAM_PARALLEL_CNT="parallel_cnt";
    @staticmethod
    def execute(environment,workspace_agt):
        workspace,agt,warpd=workspace_agt
        agt.take_action(workspace,environment);
        rws=neko_workspace();
        for k in warpd["warp_back"]:
            v=workspace.get(k);
            rws.add(k,copy.copy(v));
        for k in warpd["warp_back_log"]:
            rws.add_log(k,workspace.get_log(k));
        return rws;

    def warp_outbound(this,workspace:neko_workspace,warpd):
        w=neko_workspace();
        for k in warpd["warp_away"]:
            w.add(k,workspace.get(k));
        return w
    def warp_inbound(this,workspace:neko_workspace, sub_workspace:neko_workspace,warpd):
        for k in warpd["warp_back"]:
            workspace.add(k,sub_workspace.get(k));
        for k in warpd["warp_back_log"]:
            workspace.add_log(k,sub_workspace.get_log(k));
    def setup(this,param):
        super().setup(param);
        paracnt=neko_get_arg(this.PARAM_PARALLEL_CNT,param,2);
        this.pool=trd.init_process_group(backend="nccl",)

        this.agent_n = [];
        this.agent_d={};
        this.agent_m={};
        for name in param[this.PARAM_AGT_LST]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);
            this.agent_m[name]=param["warp_dict"][name];
        this.activation_vars=neko_get_arg(this.PARAM_ACT_VARS,param,"NEP_skipped_NEP");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        warp_env=environment.warp_ref();
        if(this.activation_vars is not None):
            for v in this.activation_vars:
                if(v not in workspace.inter_dict):
                    return workspace,environment; # if has nothing to do, do nothing.
        func = partial(this.execute,warp_env);

        workspaces= this.pool.map(func, [(this.warp_outbound(workspace,this.agent_m[n]) ,this.agent_d[n],this.agent_m[n]) for n in this.agent_n]);
        for n,w in zip(this.agent_n,workspaces):
            this.warp_inbound(workspace,w,this.agent_m[n]);
        return workspace,environment;
    @classmethod
    def append_agent_to_cfg(this,cfg,localname,subagt):
        cfg["params"][this.PARAM_AGT_LST].append(localname);
        cfg["params"][localname]=subagt;
        return cfg;

class neko_keyword_selective_execution_agent(neko_abstract_sync_agent):
    def setup(this,param):
        this.agent_n = [];
        this.agent_d={};
        for name in param["agent_list"]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);
        this.selector_name=param["selector_name"];
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        n=workspace.inter_dict[this.selector_name];
        this.agent_d[n].take_action(workspace,environment);
