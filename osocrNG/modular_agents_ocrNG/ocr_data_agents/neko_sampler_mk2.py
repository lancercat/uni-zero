import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from osocrNG.names import default_ocr_variable_names as dvn

# this one does not make labels. nor will it handle maxT
# just making prototypes.
# the maxT shit is not handled with the routing masks.
# Or you can build a post processor or whatever.
# eventually, mvn moves out, not here not now.

class neko_label_sampler_agent_mk2(neko_module_wrapping_agent):
    INPUT_label_name = dvn.raw_label_name;
    OUTPUT_tdict_name = dvn.tdict_name;
    OUTPUT_gtdict_name = dvn.gtdict_name;
    OUTPUT_weightlabel_name = dvn.proto_label_name;
    OUTPUT_gplabel_name = dvn.global_proto_label_name;
    OUTPUT_tensor_proto_img_name = dvn.tensor_proto_img_name;
    OUTPUT_proto_utf_name="proto_utf_name"; # a list of utf8 strings corresponding to the tensor_proto_image,  case sensitive.
    MOD_sampler_name = "sampler_name";
    MOD_protomvn = "protomvn";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.label = this.register_input(this.INPUT_label_name, iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_tdict_name, iocvt_dict);
        this.gtdict = this.register_output(this.OUTPUT_gtdict_name, iocvt_dict);
        this.plabel = this.register_output(this.OUTPUT_weightlabel_name, iocvt_dict);
        this.gplabel = this.register_output(this.OUTPUT_gplabel_name, iocvt_dict);
        this.tensor_proto_img_name = this.register_output(this.OUTPUT_tensor_proto_img_name, iocvt_dict);
        this.proto_utf_name=this.register_output(this.OUTPUT_proto_utf_name,iocvt_dict);
        this.sampler = this.register_mod(this.MOD_sampler_name, modcvt_dict);
        this.protomvn = this.register_mod(this.MOD_protomvn, modcvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        text_label = workspace.get(this.label);
        normprotos,proto_utfs, plabel, gplabel, tdict, gtdict = \
            environment.module_dict[this.sampler].sample_charset_by_text(
                text_label, use_sp=False, device=environment.module_dict[this.protomvn].device());
        normprotos = environment.module_dict[this.mnames.protomvn](normprotos);
        gplabel = gplabel.to(normprotos.device);
        workspace.add(this.plabel, plabel.to(normprotos.device));
        workspace.add(this.tdict, tdict);
        workspace.add(this.gtdict, gtdict);
        workspace.add(this.gplabel,gplabel);
        workspace.add(this.tensor_proto_img_name,normprotos);
        workspace.add(this.proto_utf_name,proto_utfs);
        return workspace, environment;
def get_neko_label_sampler_agent_mk2(label_name_name,
gplabel_name_name,gtdict_name_name,plabel_name_name,tdict_name_name,tensor_proto_img_name_name,proto_utf_name,
protomvn_name,sampler_name_name):
    engine = neko_label_sampler_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_label_name: label_name_name, engine.OUTPUT_gplabel_name: gplabel_name_name, engine.OUTPUT_gtdict_name: gtdict_name_name, engine.OUTPUT_weightlabel_name: plabel_name_name, engine.OUTPUT_tdict_name: tdict_name_name, engine.OUTPUT_tensor_proto_img_name: tensor_proto_img_name_name,engine.OUTPUT_proto_utf_name:proto_utf_name}, "modcvt_dict": {engine.MOD_protomvn: protomvn_name, engine.MOD_sampler_name: sampler_name_name}}}

class neko_label_sampler_agent_get_neko_label_sampler_agent_mk2_curriculum(neko_label_sampler_agent_mk2):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(workspace.epoch_idx==0 and workspace.batch_idx%5000==0 and environment.module_dict[this.sampler].max_batch_size<512):
            environment.module_dict[this.sampler].max_batch_size+=16;

        return super().take_action(workspace,environment);

def get_neko_label_sampler_agent_get_neko_label_sampler_agent_mk2_curriculum(label_name_name,
gplabel_name_name,gtdict_name_name,plabel_name_name,tdict_name_name,tensor_proto_img_name_name,
protomvn_name,sampler_name_name):
    engine = neko_label_sampler_agent_get_neko_label_sampler_agent_mk2_curriculum;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_label_name: label_name_name, engine.OUTPUT_gplabel_name: gplabel_name_name, engine.OUTPUT_gtdict_name: gtdict_name_name, engine.OUTPUT_weightlabel_name: plabel_name_name, engine.OUTPUT_tdict_name: tdict_name_name, engine.OUTPUT_tensor_proto_img_name: tensor_proto_img_name_name}, "modcvt_dict": {engine.MOD_protomvn: protomvn_name, engine.MOD_sampler_name: sampler_name_name}}}



if __name__ == '__main__':
    print(neko_label_sampler_agent_mk2.get_default_configuration_scripts());