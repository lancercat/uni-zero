from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf

class cls_loss_agent_mk2(neko_module_wrapping_agent):
    INPUT_cls_label_name = "cls_label_name";
    INPUT_cls_logit_name = "cls_logit_name";
    OUTPUT_ocr_loss_name = "loss_name";
    MOD_osocr_loss_mod_name = "osocr_loss_mod_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.clsloss = this.register_mod(this.MOD_osocr_loss_mod_name, modcvt_dict);
        this.clslabel = this.register_input(this.INPUT_cls_label_name, iocvt_dict);
        this.predcls = this.register_input(this.INPUT_cls_logit_name, iocvt_dict);
        this.lossname = this.register_output(this.OUTPUT_ocr_loss_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ocrloss= environment.module_dict[this.clsloss](workspace.get(this.predcls),workspace.get(this.clslabel));
        workspace.add_loss(this.lossname,ocrloss);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   cls_label_name, cls_logit_name,
                   ocr_loss_name,
                   osocr_loss_mod_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_cls_label_name: cls_label_name, cls.INPUT_cls_logit_name: cls_logit_name,
                           cls.OUTPUT_ocr_loss_name: ocr_loss_name},
            "modcvt_dict": {cls.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}


def get_cls_loss_agent_mk2(cls_label_name,cls_logit_name,
ocr_loss_name,
osocr_loss_mod_name,
):
    engine = cls_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name, engine.INPUT_cls_logit_name: cls_logit_name, engine.OUTPUT_ocr_loss_name: ocr_loss_name}, "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}



class per_inst_ocr_cls_loss_agent_mk2(cls_loss_agent_mk2):
    INPUT_cls_mapping_name = "cls_mapping_name";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.cls_mapping = this.register_input(this.INPUT_cls_mapping_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss = environment.module_dict[this.clsloss](
            workspace.get(this.predcls),workspace.get(this.clslabel),workspace.get(this.cls_mapping));
        workspace.add(this.lossname, per_inst_ocrloss);
        return workspace, environment;
class cls_loss_agent_mk2_debugger(neko_module_wrapping_agent):
    INPUT_cls_label_name = "cls_label_name";
    INPUT_cls_mapping_name = "cls_mapping_name";
    INPUT_tdict="tdict";
    INPUT_plabel="plabel";
    INPUT_proto_vec="proto";
    INPUT_logits="logits";
    OUTPUT_debug_text_name="cls_debug_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.cls_label_name = this.register_input(this.INPUT_cls_label_name, iocvt_dict);
        this.cls_mapping_name = this.register_input(this.INPUT_cls_mapping_name, iocvt_dict);
        this.logits = this.register_input(this.INPUT_logits, iocvt_dict);
        this.plabel = this.register_input(this.INPUT_plabel, iocvt_dict);
        this.proto_vec = this.register_input(this.INPUT_proto_vec, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.debug_text_name = this.register_output(this.OUTPUT_debug_text_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        cls_label_name = workspace.get(this.cls_label_name);
        cls_mapping_name = workspace.get(this.cls_mapping_name);
        logits = workspace.get(this.logits);
        plabel = workspace.get(this.plabel);
        proto_vec = workspace.get(this.proto_vec);
        tdict = workspace.get(this.tdict);
        jt=[tdict[cls_label_name[i].item()] for i in range(len(cls_label_name))];
        lbp=trnf.cross_entropy(logits,cls_label_name,reduction="none").detach().cpu().numpy();
        pred=[tdict[i.item()] for i in logits.argmax(dim=-1)];
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   cls_label_name, cls_mapping_name, logits, plabel, proto_vec, tdict,
                   debug_text_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_cls_label_name: cls_label_name, cls.INPUT_cls_mapping_name: cls_mapping_name,
                           cls.INPUT_logits: logits, cls.INPUT_plabel: plabel, cls.INPUT_proto_vec: proto_vec,
                           cls.INPUT_tdict: tdict, cls.OUTPUT_debug_text_name: debug_text_name}, "modcvt_dict": {}}}


def get_per_inst_ocr_cls_loss_agent_mk2(cls_label_name,cls_logit_name,cls_mapping_name,
ocr_loss_name,
osocr_loss_mod_name,
):
    engine = per_inst_ocr_cls_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name, engine.INPUT_cls_logit_name: cls_logit_name, engine.INPUT_cls_mapping_name: cls_mapping_name, engine.OUTPUT_ocr_loss_name: ocr_loss_name}, "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}


if __name__ == '__main__':
    cls_loss_agent_mk2_debugger.print_default_setup_scripts();

