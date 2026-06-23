import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNK

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
# well no sp tokens added, no length tensor added in this one.
# this will assume the thing being flattened
# if you want voodoo, make new agents

# ofc we can just, say do the replacement on translation--- however keeping a tab on this always help if we want more control over reward computing,,,
class neko_missing_to_unk(ama):
    INPUT_sam_utf="sam_utf"; # list of list of tokens.
    INPUT_tdict = "tdict";
    OUTPUT_sam_utf_wunk="sam_utf_with_unk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.sam_utf = this.register_input(this.INPUT_sam_utf, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.sam_utf_wunk = this.register_output(this.OUTPUT_sam_utf_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        sam_utf = workspace.get(this.sam_utf);
        tdict = workspace.get(this.tdict);
        outtok=[];
        for s in sam_utf:
            stok=[t if t in tdict else tUNK for t in s]
            outtok.append(stok);
        workspace.add(this.sam_utf_wunk,outtok)
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   sam_utf, tdict,
                   sam_utf_wunk
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_sam_utf: sam_utf, cls.INPUT_tdict: tdict, cls.OUTPUT_sam_utf_wunk: sam_utf_wunk},
            "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_missing_to_unk.print_default_setup_scripts()