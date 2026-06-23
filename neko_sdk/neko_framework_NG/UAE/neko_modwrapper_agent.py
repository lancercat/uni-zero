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

class neko_module_wrapping_agent(neko_abstract_sync_agent):

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        pass;
    def set_etc(this,param):
        pass;
    def setup(this,param):
        this.input_dict = {};
        this.output_dict = {};
        this.omods = {};
        this.mnames = {};
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

class neko_simple_action_module_wrapping_agent_1i1o(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input="NEP_unconfigured_NEP";
        this.mod = "NEP_unconfigured_NEP";
        this.output = "NEP_unconfigured_NEP";
        fatal("this is a virtual base class, you need to set the input, mod, and output");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.output, environment(this.mod,workspace.get(this.input)));
        return workspace;
class neko_simple_action_module_wrapping_agent_1i1o_functional(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_input="input";
    OUTPUT_output = "output";

    def op(this,input):
        fatal("this is a virtual base class, you need to set the input, op, and output");
        return input;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_input, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        pass;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.output, this.op(workspace.get(this.input)));
        return workspace;
    @classmethod
    def get_agtcfg(cls,
                   input,
                   output,
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_input: input, cls.OUTPUT_output: output},
                                         "modcvt_dict": {}}}


class neko_simple_action_module_wrapping_agent_1i1o_basic(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_input="input";
    OUTPUT_output = "output";
    MOD_mod = "mod";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_input, iocvt_dict);
        this.output = this.register_output(this.OUTPUT_output, iocvt_dict);
        this.mod = this.register_mod(this.MOD_mod, modcvt_dict);
        pass;

    @classmethod
    def get_agtcfg(cls,
                   input,
                   output,
                   mod
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_input: input, cls.OUTPUT_output: output},
                                         "modcvt_dict": {cls.MOD_mod: mod}}}

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.output, environment(this.mod,workspace.get(this.input)));
        return workspace;

if __name__ == '__main__':
    neko_simple_action_module_wrapping_agent_1i1o_basic.print_default_setup_scripts()