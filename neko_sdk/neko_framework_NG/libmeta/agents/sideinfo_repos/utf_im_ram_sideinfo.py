import os.path

import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.agents.sideinfo_repos.abstract_sinfo_holder_agent import abstract_sideinfo_index_agent
# load from precached images and fetch with utf
# More versions will come if needed
class utf_precached_ram_sideinfo_one(abstract_sideinfo_index_agent):

    def set_etc(this, params):
        this.mod_path = neko_get_arg(this.PARAM_mod_path, params);
        this.cache=torch.load(os.path.join(this.mod_path,"glyph_repo.pt"),weights_only=False);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        utf = workspace.get(this.utf);
        workspace.add(this.sideinfo, [this.cache[i] for i in utf]);
        workspace.add(this.sideinfo_utf, [i for i in utf]);
        workspace.alias(this.tdict,this.sideinfo_tdict);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   utf, tdict,
                   sideinfo, sideinfo_utf, sideinfo_tdict,
                   mod_path
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_utf: utf,cls.INPUT_tdict:tdict, cls.OUTPUT_sideinfo: sideinfo, cls.OUTPUT_sideinfo_utf: sideinfo_utf, cls.OUTPUT_sideinfo_tdict: sideinfo_tdict},
            cls.PARAM_mod_path: mod_path,  "modcvt_dict": {}}}


class utf_precached_im_ram_sideinfo_one(utf_precached_ram_sideinfo_one):
    @classmethod
    def get_agtcfg(cls,
                   utf,tdict,
                   sideinfo, sideinfo_utf,sideinfo_tdict,
                   root
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_utf: utf,cls.INPUT_tdict:tdict, cls.OUTPUT_sideinfo: sideinfo, cls.OUTPUT_sideinfo_utf: sideinfo_utf, cls.OUTPUT_sideinfo_tdict: sideinfo_tdict},
            cls.PARAM_mod_path: root, "modcvt_dict": {}}}



class utf_precached_text_ram_sideinfo_one(utf_precached_ram_sideinfo_one):
    DFT_PATH="text_templates.pt";
if __name__ == '__main__':
    utf_precached_ram_sideinfo_one.print_default_setup_scripts()