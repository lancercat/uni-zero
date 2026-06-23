import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent,neko_simple_action_module_wrapping_agent_1i1o
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from torch.nn import functional as trnf

# well if you need fancy shit you can always create another one.
class simple_spatial_embedding(neko_simple_action_module_wrapping_agent_1i1o):
    INPUT_feature_maps="feats";
    OUTPUT_feature_maps="last_feat";
    MOD_se="se_mod"
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input_list(this.INPUT_feature_maps,iocvt_dict);
        this.output=this.register_output_list(this.OUTPUT_feature_maps, iocvt_dict);
        this.mod=this.register_mod(this.MOD_se,modcvt_dict);

def get_simple_spatial_embedding(in_feature_maps_name,out_feature_maps_name, se_name):
    engine = simple_spatial_embedding;
    return {
        "agent": engine,
        "params": {
            "iocvt_dict":
                {
                    engine.INPUT_feature_maps: in_feature_maps_name,
                    engine.OUTPUT_feature_maps: out_feature_maps_name
                },
            "modcvt_dict": {
                engine.MOD_se: se_name
            }
        }
    };
class transformed_spatial_embedding(neko_module_wrapping_agent):
    INPUT_feature_maps="feats";
    INPUT_transform_grids="transform_grids";
    OUTPUT_embedded_feature_maps="last_feat";
    MOD_se="se_mod";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feature_maps = this.register_input(this.INPUT_feature_maps, iocvt_dict);
        this.transform_grids = this.register_input(this.INPUT_transform_grids, iocvt_dict);
        this.embedded_feature_maps = this.register_output(this.OUTPUT_embedded_feature_maps, iocvt_dict);
        this.se = this.register_mod(this.MOD_se, modcvt_dict);
        pass;



    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        feature_maps = workspace.get(this.feature_maps);
        transform_grids = workspace.get(this.transform_grids);
        raw_ses=environment.module_dict[this.se]();
        allout=[];
        for i in range(len(raw_ses)):
            se= trnf.grid_sample(raw_ses[i].repeat([transform_grids.shape[0],1,1,1]),
                                 trnf.interpolate(transform_grids,(feature_maps[i].shape[2],feature_maps[i].shape[3])).permute(0, 2, 3, 1),
                                 mode="bilinear");
            allout.append(torch.cat([se.expand(feature_maps[i].shape[0],-1,-1,-1),feature_maps[i]],dim=1));

        workspace.add(this.embedded_feature_maps,allout);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   feature_maps, transform_grids,
                   embedded_feature_maps,
                   se
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_feature_maps: feature_maps, cls.INPUT_transform_grids: transform_grids,
                           cls.OUTPUT_embedded_feature_maps: embedded_feature_maps}, "modcvt_dict": {cls.MOD_se: se},
            }}

def get_transformed_spatial_embedding(
        feature_maps, transform_grids,
        embedded_feature_maps,
        se,
):
    engine = transformed_spatial_embedding;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_feature_maps: feature_maps, engine.INPUT_transform_grids: transform_grids,
                       engine.OUTPUT_embedded_feature_maps: embedded_feature_maps},
        "modcvt_dict": {engine.MOD_se: se}}}


if __name__ == '__main__':
    transformed_spatial_embedding.print_default_setup_scripts();
