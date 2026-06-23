# when the thing only have one modality of side information....
import os.path

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.modules.side_info.abstract_mod import neko_utf_dict_meta_repo_mk2
from neko_sdk.neko_framework_NG.libmeta.file_names import FN

from neko_sdk.neko_framework_NG.libmeta.modules.side_info.tdict_repo import neko_tdict_repo

class neko_agentic_sideinfo_repo_id(ama):
    OUTPUT_tdict="tdict";
    OUTPUT_side_info="sideinfo";
    OUTPUT_side_info_utf="utf";

    PARAM_root="meta_path";
    def set_up_sideinfo(this):
        this.side_info_repo = neko_utf_dict_meta_repo_mk2(
            {neko_utf_dict_meta_repo_mk2.PARAM_root:os.path.join(this.root),
             neko_utf_dict_meta_repo_mk2.PARAM_name:FN.ID_META}
        )

        pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.side_info = this.register_output(this.OUTPUT_side_info, iocvt_dict);
        this.side_info_utf = this.register_output(this.OUTPUT_side_info_utf, iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_tdict, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.root = neko_get_arg(this.PARAM_root, params);
        this.tdict_factory=neko_tdict_repo({
            neko_tdict_repo.PARAM_root:this.root
        }
        )
        this.set_up_sideinfo();
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        workspace.add(this.tdict,this.tdict_factory());
        si,utf=this.side_info_repo();
        workspace.add(this.side_info,si);
        workspace.add(this.side_info_utf,utf); # i think with this we do not have to have the sample utf....
                                                                        # since plabel will simply be trascribed from tdict, wrt utf....
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   side_info, side_info_utf, tdict,
                   root
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.OUTPUT_side_info: side_info, cls.OUTPUT_side_info_utf: side_info_utf,
                           cls.OUTPUT_tdict: tdict}, cls.PARAM_root: root, "modcvt_dict": {}}}

    

