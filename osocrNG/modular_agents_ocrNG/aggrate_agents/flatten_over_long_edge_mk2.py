from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred,neko_flatten_NG_idx_mapping
import torch
import torch.nn.functional as trnf;

# it's okay if it does not cover all situations.
# we have another dan-based engine after all.
class neko_flatten_against_long_edge_agent_mk2_naive(neko_module_wrapping_agent):
    INPUT_feat_map="feat_map"
    OUTPUT_feat_seq_name="feat_seq_name";
    PARAM_hparz="hparz";
    PARAM_wpatz="wparz";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat_map = this.register_input(this.INPUT_feat_map, iocvt_dict);
        this.feat_seq_name = this.register_output(this.OUTPUT_feat_seq_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.hparz = neko_get_arg(this.PARAM_hparz, params);
        this.wpatz = neko_get_arg(this.PARAM_wpatz, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        feat_map_ = workspace.get(this.feat_map)[-1];
        if (this.hparz*this.wpatz>1):
            N,C,H,W=feat_map_.shape;
            feat_map= feat_map_.reshape([N,C,H//this.hparz,this.hparz,W//this.wpatz,this.wpatz]).permute(0,3,5,1,2,4).reshape(N,this.hparz*this.wpatz,C,H//this.hparz,W//this.wpatz)

        else:
            feat_map=feat_map_.unsqueeze(1);
        _,_, _, w, h = feat_map.shape;
        if(w>=h):
            ff=feat_map.mean(-1);
        else:
            ff=feat_map.mean(-2);
        #N P C T -> N T P C
        workspace.add(this.feat_seq_name,ff.permute(0,3,1,2));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   feat_map,
                   feat_seq_name,
                   hparz=2, wpatz=2
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_feat_map: feat_map, cls.OUTPUT_feat_seq_name: feat_seq_name},
                           cls.PARAM_hparz: hparz, cls.PARAM_wpatz: wpatz, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_flatten_against_long_edge_agent_mk2_naive.print_default_setup_scripts()