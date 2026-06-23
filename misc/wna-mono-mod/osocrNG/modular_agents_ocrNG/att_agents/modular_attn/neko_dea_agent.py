# Dynamic_query_gen
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_basic_temporal_attention_och_global_observation(neko_module_wrapping_agent):
    INPUT_feats = "feats";
    INPUT_global_observation = "global_observation";
    OUTPUT_raw_masks = "amsks";
    MOD_attmod = "attm";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input = this.register_input(this.INPUT_feats, iocvt_dict);
        this.gobs=this.register_input(this.INPUT_global_observation,iocvt_dict);
        this.mod = this.register_mod(this.MOD_attmod, modcvt_dict);
        this.output = this.register_output(this.OUTPUT_raw_masks, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):

        workspace.add(this.output, environment(this.mod,workspace.get(this.input),workspace.get(this.gobs)));
        return workspace;
def get_neko_basic_temporal_attention_och_global_observation(
        feats_name,global_observation,
        masks_name,
        attmod_name):
    engine = neko_basic_temporal_attention_och_global_observation;
    return {
        "agent": engine,
        "params": {
            "iocvt_dict": {
                engine.INPUT_feats: feats_name,
                engine.INPUT_global_observation: global_observation,
                engine.OUTPUT_raw_masks: masks_name
            },
            "modcvt_dict": {
                engine.MOD_attmod: attmod_name
            }
        }
    }

