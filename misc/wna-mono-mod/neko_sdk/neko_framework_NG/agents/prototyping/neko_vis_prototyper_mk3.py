import torch
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace

class neko_vis_prototyper_agent_mk3(neko_module_wrapping_agent):
    MOD_fe_name="prototyper_name";
    MOD_att_name="att_name";
    MOD_aggr_name="aggr_name";
    INPUT_tensor_proto_img_name="tensor_proto_img_name";
    OUTPUT_raw_tensor_proto_vec_name="raw_tensor_proto_vec_name"; # pre normalization. Used for ssl.
    OUTPUT_tensor_proto_vec_name="tensor_proto_vec_name";
    PARAM_possible_rotation="possible_rotation";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tensor_proto_img_name = this.register_input(this.INPUT_tensor_proto_img_name, iocvt_dict);
        this.tensor_proto_vec_name = this.register_output(this.OUTPUT_tensor_proto_vec_name, iocvt_dict);
        this.raw_tensor_proto_vec_name=this.register_output(this.OUTPUT_raw_tensor_proto_vec_name,iocvt_dict);
        this.aggr_name = this.register_mod(this.MOD_aggr_name, modcvt_dict);
        this.att_name = this.register_mod(this.MOD_att_name, modcvt_dict);
        this.fe_name = this.register_mod(this.MOD_fe_name, modcvt_dict);
        pass;
    def make_protos(this,environment,proto_ims):
        proto_featmaps = environment.module_dict[this.fe_name](proto_ims);
        proto_att=environment.module_dict[this.att_name](proto_featmaps);
        proto_raw=environment.module_dict[this.aggr_name](proto_featmaps[-1],proto_att).squeeze(1); # remove T
        protos = trnf.normalize(proto_raw, p=2, dim=-1);
        return protos,(proto_featmaps,proto_att,proto_raw);

    def set_etc(this, params):
        this.possible_rotation = neko_get_arg(this.PARAM_possible_rotation, params,[]);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        protos,(proto_featmaps,proto_att,proto_raw) = this.make_protos(environment, workspace.inter_dict[this.tensor_proto_img_name]);
        workspace.add(this.tensor_proto_vec_name,protos);
        workspace.add(this.raw_tensor_proto_vec_name,proto_raw);
        workspace.inter_dict[this.tensor_proto_vec_name] = protos;
        return workspace, environment

    @classmethod
    def get_agtcfg(cls,
                   tensor_proto_img_name,
                    tensor_proto_vec_name,raw_tensor_proto_vec_name,
                   aggr_name, att_name, fe_name
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_tensor_proto_img_name: tensor_proto_img_name,
                                                        cls.OUTPUT_raw_tensor_proto_vec_name:raw_tensor_proto_vec_name,
                                                        cls.OUTPUT_tensor_proto_vec_name: tensor_proto_vec_name},
                                         "modcvt_dict": {cls.MOD_aggr_name: aggr_name, cls.MOD_att_name: att_name,
                                                         cls.MOD_fe_name: fe_name}}}


class neko_vis_prototyper_agent_mk3_rot(neko_vis_prototyper_agent_mk3):

    OUTPUT_rotated_tensor_proto_vec_name="rotated_tensor_proto_vec_name";
    OUTPUT_raw_rotated_tensor_proto_vec_name="raw_rotated_tensor_proto_vec_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.rotated_tensor_proto_vec_name = this.register_output(this.OUTPUT_rotated_tensor_proto_vec_name,
                                                                  iocvt_dict);
        this.raw_rotated_tensor_proto_vec_name = this.register_output(this.OUTPUT_raw_rotated_tensor_proto_vec_name,
                                                                  iocvt_dict);

    def set_etc(this, params):
        this.possible_rotation = neko_get_arg(this.PARAM_possible_rotation, params,[]);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        protos, (proto_featmaps, proto_att, proto_raw) = this.make_protos(environment, workspace.inter_dict[
            this.tensor_proto_img_name]);
        workspace.add(this.tensor_proto_vec_name, protos);
        workspace.add(this.raw_tensor_proto_vec_name, proto_raw);
        workspace.inter_dict[this.tensor_proto_vec_name] = protos;
        if (len(this.possible_rotation)):
            workspace.add(this.rotated_tensor_proto_vec_name, {0: protos}) ;
            workspace.add(this.raw_rotated_tensor_proto_vec_name, {0: proto_raw}) ;

            for k in this.possible_rotation:
                if (k == 0):
                    continue;
                workspace.inter_dict[this.rotated_tensor_proto_vec_name][k],(_, __, workspace.inter_dict[this.raw_rotated_tensor_proto_vec_name][k])= this.make_protos(environment, torch.rot90(
                    workspace.inter_dict[this.tensor_proto_img_name], k=k, dims=[2, 3]))
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tensor_proto_img_name,
                   tensor_proto_vec_name, raw_tensor_proto_vec_name,
                   rotated_tensor_proto_vec_name,raw_rotated_tensor_proto_vec_name,
                   aggr_name, att_name, fe_name,
                   possible_rotation
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_tensor_proto_img_name: tensor_proto_img_name,
                                                        cls.OUTPUT_raw_rotated_tensor_proto_vec_name: raw_rotated_tensor_proto_vec_name,
                                                        cls.OUTPUT_raw_tensor_proto_vec_name: raw_tensor_proto_vec_name,
                                                        cls.OUTPUT_rotated_tensor_proto_vec_name: rotated_tensor_proto_vec_name,
                                                        cls.OUTPUT_tensor_proto_vec_name: tensor_proto_vec_name},
                                         "modcvt_dict": {cls.MOD_aggr_name: aggr_name, cls.MOD_att_name: att_name,
                                                         cls.MOD_fe_name: fe_name},
                                         cls.PARAM_possible_rotation: possible_rotation}}


if __name__ == '__main__':
    neko_vis_prototyper_agent_mk3_rot.print_default_setup_scripts()