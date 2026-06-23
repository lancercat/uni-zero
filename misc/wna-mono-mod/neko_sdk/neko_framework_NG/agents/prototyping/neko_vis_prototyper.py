import torch
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.neko_framework_NG.names import default_variable_names as dvn

class neko_vis_prototyper_agent(neko_module_wrapping_agent):
    MOD_prototyper_name="prototyper_name";
    INPUT_tensor_proto_img_name=dvn.tensor_proto_img_name;
    OUTPUT_tensor_proto_vec_name=dvn.tensor_proto_vec_name;
    OUTPUT_rotated_tensor_proto_vec_name=dvn.rotated_tensor_proto_vec_name;
    def set_etc(this,param):
        this.possible_rotation=neko_get_arg("possible_rotation",param,[]);
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.prototyper_name=this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.protoimage=this.register(this.INPUT_tensor_proto_img_name,iocvt_dict,this.input_dict);
        this.protovector=this.register(this.OUTPUT_tensor_proto_vec_name,iocvt_dict,this.output_dict);
        this.rotated_proto_vecs=this.register(this.OUTPUT_rotated_tensor_proto_vec_name,iocvt_dict,this.output_dict,"NEP_skipped_NEP");

    def make_protos(this,environment,proto_ims):
        protos = environment.module_dict[this.prototyper_name](proto_ims)
        protos = trnf.normalize(protos, p=2, dim=-1);
        return protos;

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        protos=this.make_protos(environment,workspace.inter_dict[this.protoimage]);
        workspace.inter_dict[this.protovector]=protos;
        if(len(this.possible_rotation)):
            workspace.inter_dict[this.rotated_proto_vecs]={0:protos};
            for k in this.possible_rotation:
                if(k==0):
                    continue;
                workspace.inter_dict[this.rotated_proto_vecs][k]=this.make_protos(environment,torch.rot90(workspace.inter_dict[this.protoimage], k=k, dims=[2, 3]))
        return workspace,environment

def get_prototyping_agent(tensor_proto_img_name,tensor_proto_vec_name, prototyper_name):
    return {
        "agent":neko_vis_prototyper_agent,
        "params":{
            "iocvt_dict": {
                "tensor_proto_img_name":tensor_proto_img_name,
                "tensor_proto_vec_name":tensor_proto_vec_name,
            },
            "modcvt_dict": {
                "prototyper_name": prototyper_name,
            }
        }
    }