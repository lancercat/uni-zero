# when the thing only have one modality of side information....
# let's make meta always cacheable -- unless we have situation otherwise, e.g. too big knowledge base to fit in ram.
# tdict : {
# id-> utf # used for decode
# utf-> id # used for encode
# }
# note more than 1 utf can be mapped to the same id.
# an utf may or may not exist for all modalities.
# but the main utf (tdict[id]) needs to exist for all ids.
# this means each modality *may* have their own plabels. --- which makes sense.
# A in comic sans can have the same linguistic effect to A in times new roman---
# Well that's not really true, but I put *may*. Did I not?

import os.path

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.file_names import FN

from neko_sdk.neko_framework_NG.libmeta.modules.side_info.tdict_repo import load_tdict, neko_tdict_repo
# in 33x we modularize modalities of sideinfo
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.libmeta.agents.utils import mk_meta,sorted_utf

# in 32x and above meta will always be accompanied by a hash--- so that everything with a memory can track if it has been changed.

class neko_meta_root(ama):
    OUTPUT_meta_path="meta_path";
    OUTPUT_meta_hash="meta_hash";
    PARAM_root="root";
    PARAM_file_name="file_name"; # don't worry if you are not doing magic

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.meta_path = this.register_output(this.OUTPUT_meta_path, iocvt_dict);
        this.meta_hash = this.register_output(this.OUTPUT_meta_hash, iocvt_dict);

        pass;

    def set_etc(this, params):
        this.path=os.path.join(
            neko_get_arg(this.PARAM_root, params),
            neko_get_arg(this.PARAM_file_name, params,FN.TDICT));
        this.hash=this.hash_meta(this.path);
        pass;
    def hash_meta(this,path):
        return path
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        workspace.add(this.meta_path,this.path);
        workspace.add(this.meta_hash,this.hash);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   meta_path,meta_hash,
                   root, file_name=FN.TDICT
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.OUTPUT_meta_path: meta_path,cls.OUTPUT_meta_hash:meta_hash}, cls.PARAM_file_name: file_name,
                           cls.PARAM_root: root, "modcvt_dict": {}}}


# it is decoupled this way bcs it does not bite :)
# during testing we don't want to load anything from disk--or from ram more than once.
# if we have rabbiting sideinfo hadling, the tdict is still not bijection-- bcs we need that for sampling
# tdict-> sampling-> arm sideinfo->|dataloadr|pipe|net|-> decorate (sptoken/eos)-> [mkplabels][mktlabels] perhead
class neko_agentic_sideinfo_repo_dyna(ama):
    INPUT_meta_path="meta_path";
    INPUT_meta_hash="meta_hash";

    OUTPUT_tdict="tdict";
    # we have ids. but we will give the ids from the sampler.
    # for testing w/o sampling we will have another agent to do that


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.meta_path = this.register_input(this.INPUT_meta_path, iocvt_dict);
        this.meta_hash = this.register_input(this.INPUT_meta_hash, iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_tdict, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.known_tdict={};

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        meta_root = workspace.get(this.meta_path);
        meta_hash=workspace.get(this.meta_hash);
        if(meta_root not in this.known_tdict):
            tdict=load_tdict(meta_root);
            this.known_tdict[meta_hash]=tdict;
        workspace.add(this.tdict,this.known_tdict[meta_hash]);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   meta_path,meta_hash,
                   tdict
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_meta_path: meta_path,cls.INPUT_meta_hash:meta_hash, cls.OUTPUT_tdict: tdict},
                           "modcvt_dict": {}}}
class neko_mksrted_utf(ama):
    INPUT_tdict="tdict";
    OUTPUT_utf="utf";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.utf = this.register_output(this.OUTPUT_utf, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        utf=sorted_utf(tdict);
        workspace.add(this.utf,utf);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tdict,
                   utf
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_tdict: tdict, cls.OUTPUT_utf: utf}, "modcvt_dict": {}}}



# is tdict sideinfo? i would say not intentionally so. its more like a primary key thing, or in an xml sense, DTD
class neko_agentic_tdict_repo_static:
    PARAM_root = "root";
    PARAM_file_name = "file_name";  # don't worry if you are not doing magic
    OUTPUT_meta_hash="meta_hash";
    OUTPUT_meta_path = "meta_path";
    OUTPUT_tdict="tdict";
    OUTPUT_utf="utf";

    @classmethod
    def get_agtcfg(cls,
                   meta_path,meta_hash,tdict, utf,
                   root, file_name=FN.TDICT):
        agtd=awa.empty();
        agtd=awa.append_agent_to_cfg(
            agtd,"meta_info",neko_meta_root.get_agtcfg(
                meta_path, meta_hash,
                root, file_name
            ));
        agtd=awa.append_agent_to_cfg(
            agtd, "arm_tdict", neko_agentic_sideinfo_repo_dyna.get_agtcfg(
                meta_path, meta_hash,
                tdict
            ));
        agtd=awa.append_agent_to_cfg(
            agtd,"arm_utflst", neko_mksrted_utf.get_agtcfg(
                tdict,utf
            )
        );
        return agtd;


if __name__ == '__main__':
    neko_mksrted_utf.print_default_setup_scripts();
