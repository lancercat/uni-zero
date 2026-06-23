# dumps the workspace for each single instance
# writes a hellot and is hella slow. Literally works like a coredump
# Don't ever use it to write in an SSD that holds important data!!!!
import os.path

import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.agents.debugging.neko_abstract_debugging_agent import neko_abstract_debugging_agent

class neko_workspace_dumper(neko_abstract_debugging_agent):
    DFT_saveprfx = "wsdump";
    DFT_postfx = ".pt";

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        fns,_ = this.get_items(workspace);
        ss=workspace.simple_fetch_interdict_as_subspace(this.keys);
        if(len(fns)==1):
            torch.save(ss,fns[0]);
        else:
            fatal("not impl");
        return workspace,environment
def get_neko_workspace_dumper(uid_name,keys,dstpath,saveprfx):
    engine = neko_workspace_dumper;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_uid: uid_name}, engine.PARAM_dstpath: dstpath, engine.PARAM_keys: keys, engine.PARAM_saveprfx: saveprfx, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_workspace_dumper.get_default_configuration_scripts());