
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_mvn_agent(neko_module_wrapping_agent):
    INPUT_raw_image_names="raw_image_names";
    OUTPUT_tensor_image_names="tensor_image_names";
    MOD_mvn_mod_name="mvn_mod_name";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.raw_image_names=neko_get_arg(this.INPUT_raw_image_names,iocvt_dict);
        this.output_dict.tensor_image_names=neko_get_arg(this.OUTPUT_tensor_image_names,iocvt_dict);
        this.mvn_name=this.register_mod(this.MOD_mvn_mod_name,modcvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        for i in range(len(this.input_dict.raw_image_names)):
            iname=this.input_dict.raw_image_names[i];
            oname=this.output_dict.tensor_image_names[i];
            workspace.inter_dict[oname]=environment.module_dict[this.mvn_name](workspace.inter_dict[iname]);
        return workspace,environment;

def get_neko_mvn_agent(raw_image_names,tensor_image_names,mvn_mod_name):
    engine=neko_mvn_agent;
    return {
        "agent":engine,
        "params":{
            "iocvt_dict":{
                engine.INPUT_raw_image_names:raw_image_names,
                engine.OUTPUT_tensor_image_names:tensor_image_names,
            },
            "modcvt_dict":
            {
                engine.MOD_mvn_mod_name: mvn_mod_name,
            }
        }
    }