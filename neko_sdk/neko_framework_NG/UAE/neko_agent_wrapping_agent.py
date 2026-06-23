# NG hoards a bunch of agents(mod). within a bunch of agents.
import copy
import time

import torch.cuda
# from easydict import EasyDict

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
            # print(n,"--starts--")
            this.agent_d[n].take_action(workspace,environment);
            # print(n,"--ends--")
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
    def wrap_this(cls, subagt,delay_bid="NEP_skipped_NEP",delay_eid=0,actvars=None):
        if actvars is None:
            actvars=[];
        return {
            "agent": neko_agent_wrapping_agent,
            "params": {
                cls.PARAM_AGT_LST: ["meow"],
                "meow": subagt,
                cls.PARAM_ACT_VARS:actvars,
                 cls.PARAM_disable_till_bid: delay_bid,
                 cls.PARAM_disable_till_eid:delay_eid,
            }
        }
    @classmethod
    def empty(cls,actvars=None):
        if(actvars is None):
            actvars=[];
        return  {
            "agent": cls,
            "params": {
                "agent_list": [],
                cls.PARAM_ACT_VARS: actvars
            }
        }
    @classmethod
    def append_delayed_agent_to_cfg(this,cfg,localname,subagt,delay_bid,delay_eid):
        cfg["params"][this.PARAM_AGT_LST].append(localname);
        cfg["params"][localname]=this.wrap_this(subagt,delay_bid=delay_bid,delay_eid=delay_eid);
        return cfg;
    @classmethod
    def prepend_agent_to_cfg(this, cfg, localname, subagt):
        cfg["params"][this.PARAM_AGT_LST]=[localname]+cfg["params"][this.PARAM_AGT_LST];
        cfg["params"][localname] = subagt;
        return cfg;
    def get_agt_at(this,idx):
        return this.agent_d[this.agent_n[idx]];
# don't you ever store gradient to your cache lol.
# only use this in inference. For training you
# if cache hash match, inject the cache to the dict instead of calling all the agents wrapped by it.
# if cache hash does not match, dump cache, set hash, execute agents, and build it again.
# if cache hash not exist, means constant changing values, just call all the agents and don't bother with cache, making it an awa
# note case 3 does not dump its cache. tester during training will have to inject some hash to intentionally dump the caches.
# need to be chained together to a hashing agent
# awa{
#    hashing_agt{
#          hashing_mod
#    }
#    neko_agent_wrapping_agent_cachable_cache_nograd{
#       agent1,
#       agent2....
#    }
# }

class neko_agent_wrapping_agent_cachable_cache_nograd(neko_agent_wrapping_agent):
    INPUT_CACHE_HASH_VAR="cache_hash"; # the hash of anticipated cache
    PARAM_CACHED_VARS="cached_vars";
    def setup(this,param):
        super().setup(param);
        this.cache=None;
        this.cache_hash=None;
        this.cache_hash_var=neko_get_arg(this.INPUT_CACHE_HASH_VAR,param);
        this.cached_vars=neko_get_arg(this.PARAM_CACHED_VARS,param);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # no cac
        if(this.cache_hash_var not in workspace.inter_dict):
            return super().take_action(workspace,environment);
        hash=workspace.get(this.cache_hash_var);
        if(hash==this.cache_hash):
            for k in this.cache:
                workspace.add(k,this.cache[k]);
        else:
            super().take_action(workspace, environment);
            this.cache_hash=hash;
            this.cache={};
            for k in this.cached_vars:
                this.cache[k]=workspace.get(k);
        return workspace, environment;


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
class neko_agent_wrapping_agent_nograd_timed(neko_agent_wrapping_agent_nograd):
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
                sta=time.time();
                this.agent_d[n].take_action(workspace,environment);
                end=time.time();
                print(n,"--takes--",end-sta)
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
