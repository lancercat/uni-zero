

# decorates protos and tdict

# what about part based stuff?????? (expainable stuff)

import torch
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama


class neko_meta_decor(ama):
    INPUT_tdict = "tdict";
    INPUT_utf="utf";

    OUTPUT_tdict_mod = "tdict_mod"
    OUTPUT_proto_vec_mod="proto_vec_mod";

# plabels are always generated on the fly from tdict.
# as sptokes, and unk tokens do not add extra rules and elaborates to plabels.
# modding can be casted. they will inject from the lower end (from 0 up)
# the labels can be <eos> <blank> <sta>... etc
# tokenizer will be in its own agent....
from neko_sdk.neko_framework_NG.libmeta.agents.utils import mk_id_dict


# inject centered_tokens with module.
# cannot be used to insert centerless tokens like tunk.
class generic_centered_tokens_injector(ama):
    INPUT_tdict="tdict";
    INPUT_proto_vec="proto_vec";
    INPUT_proto_utf="proto_utf";

    OUTPUT_proto_utf_mod="proto_utf_mod";
    OUTPUT_tdict_mod = "tdict_mod";
    OUTPUT_proto_vec_mod="proto_vec_mod";
    MOD_centered_meta_modders="special_embeddings";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto_utf = this.register_input(this.INPUT_proto_utf, iocvt_dict);
        this.proto_vec = this.register_input(this.INPUT_proto_vec, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.proto_utf_mod = this.register_output(this.OUTPUT_proto_utf_mod, iocvt_dict);
        this.proto_vec_mod = this.register_output(this.OUTPUT_proto_vec_mod, iocvt_dict);
        this.tdict_mod = this.register_output(this.OUTPUT_tdict_mod, iocvt_dict);
        this.centered_meta_modders = this.register_mod(this.MOD_centered_meta_modders, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        proto_utf = workspace.get(this.proto_utf);
        proto_vec = workspace.get(this.proto_vec);
        tdict = workspace.get(this.tdict);
        new_tdict,new_proto_vec,new_utf=environment(this.centered_meta_modders,tdict,proto_vec,proto_utf);
        workspace.add(this.proto_utf_mod,new_utf);
        workspace.add(this.proto_vec_mod, new_proto_vec);
        workspace.add(this.tdict_mod, new_tdict);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   proto_utf, proto_vec, tdict,
                   proto_utf_mod, proto_vec_mod, tdict_mod,
                   centered_meta_modders
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_proto_utf: proto_utf, cls.INPUT_proto_vec: proto_vec, cls.INPUT_tdict: tdict,
                           cls.OUTPUT_proto_utf_mod: proto_utf_mod, cls.OUTPUT_proto_vec_mod: proto_vec_mod,
                           cls.OUTPUT_tdict_mod: tdict_mod}, "modcvt_dict": {cls.MOD_centered_meta_modders: centered_meta_modders}}}

