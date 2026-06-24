from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent

from osocrNG.names import default_ocr_variable_names as dvn
import torch
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.visualization.show_tims import show_tims


# it does not have a module to wrap, that is.
class neko_vis_agent(neko_module_wrapping_agent):
    INPUT_tensor_img_name=dvn.tensor_image_name;
    INPUT_tensor_beacon_name = dvn.tensor_beacon_name;
    INPUT_tensor_proto_img_name=dvn.tensor_proto_img_name;
    INPUT_tensor_plabel_name=dvn.proto_label_name;
    INPUT_tdict_name=dvn.tdict_name;
    INPUT_raw_gt_text_name=dvn.raw_label_name;
    OUTPUT_raw_image="visualization_name";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.tensor_img_name=this.register(this.INPUT_tensor_img_name,iocvt_dict,this.input_dict);
        this.tensor_beacon_name=this.register(this.INPUT_tensor_beacon_name, iocvt_dict, this.input_dict);
        this.tensor_proto_img_name=this.register(this.INPUT_tensor_proto_img_name, iocvt_dict, this.input_dict);
        this.tensor_plabel_name=this.register(this.INPUT_tensor_plabel_name, iocvt_dict, this.input_dict);
        this.tdict_name=this.register(this.INPUT_tensor_img_name, iocvt_dict, this.input_dict);
        this.raw_gt_text_name=this.register(this.INPUT_tdict_name, iocvt_dict, this.input_dict);
        this.visualization_name=this.register(this.OUTPUT_raw_image,iocvt_dict,this.input_dict);
    def set_etc(this,param):
        this.mean=torch.tensor(neko_get_arg("mean",param,[127,127,127])).float().reshape([1,3,1,1]);
        this.var=torch.tensor(neko_get_arg("var",param,(127,127,127))).float().reshape(1,3,1,1);
        this.name=neko_get_arg("name",param,"debug_image");
    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        imt=workspace.inter_dict[this.tensor_img_name];
        show_tims(imt,this.name,this.mean.cpu().numpy(),this.var.cpu().numpy());
        return workspace;
