# indexes whatever embedding you have with the provided ids

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama

class neko_embedding_indexer(ama):
    INPUT_ids="ids";
    OUTPUT_embeddings="embeddings";
    MOD_embed_mgr="embedding_mgr";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.ids = this.register_input(this.INPUT_ids, iocvt_dict);
        this.embeddings = this.register_output(this.OUTPUT_embeddings, iocvt_dict);
        this.embed_mgr = this.register_mod(this.MOD_embed_mgr, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ids = workspace.get(this.ids);
        workspace.add(this.embeddings,environment(this.embed_mgr,ids));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   ids,
                   embeddings,
                   embed_mgr
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_ids: ids, cls.OUTPUT_embeddings: embeddings},
                                         "modcvt_dict": {cls.MOD_embed_mgr: embed_mgr}}}


if __name__ == '__main__':
    neko_embedding_indexer.print_default_setup_scripts()