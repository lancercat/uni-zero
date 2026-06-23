from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# You need no loss during evaluation, so we give the choice of skipping back to you :-)
class ocr_loss_agent(neko_module_wrapping_agent):
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        # Yup, classification of length and recognition, nothing more nothing less.
        # This stub sticks to basic.
        # If you need more, go grab a regularization subroutine
        this.mnames.clsloss = neko_get_arg("osocr_loss_mod_name", modcvt_dict);
        this.input_dict.clslabel = neko_get_arg("cls_label_name", iocvt_dict, "cls_label");
        this.input_dict.lenlabel = neko_get_arg("len_label_name", iocvt_dict, "len_label");
        this.input_dict.clslog = neko_get_arg("cls_logit_name", iocvt_dict, "cls_logit");
        this.input_dict.lenlog = neko_get_arg("len_logit_name", iocvt_dict, "len_logit");
        this.output_dict.ocrloss = neko_get_arg("ocr_loss_name", iocvt_dict, "ocr_loss");
        this.loss_name = neko_get_arg("loss_name", iocvt_dict, "loss");

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ocrloss, terms = environment.module_dict[this.mnames.clsloss](
            workspace.inter_dict[this.input_dict.clslog],
            workspace.inter_dict[this.input_dict.lenlog],
            workspace.inter_dict[this.input_dict.clslabel].detach(),
            workspace.inter_dict[this.input_dict.lenlabel].detach(),
        );
        workspace.objdict[this.output_dict.ocrloss] = ocrloss;
        workspace.logdict[this.output_dict.ocrloss] = terms;

        return workspace;


def get_ocr_loss_agent(cls_label_name, cls_logit_name, len_logit_name, len_label_name, ocr_loss_name,
                       osocr_loss_mod_name):
    return {
        "agent": ocr_loss_agent,
        "params": {
            "iocvt_dict": {
                "cls_label_name": cls_label_name,
                "cls_logit_name": cls_logit_name,
                "len_logit_name": len_logit_name,
                "len_label_name": len_label_name,
                "ocr_loss_name": ocr_loss_name,
            },
            "modcvt_dict": {
                "osocr_loss_mod_name": osocr_loss_mod_name,
            }
        }
    }


class ocr_loss_agent_mk2(neko_module_wrapping_agent):
    INPUT_cls_label_name = "cls_label_name";
    INPUT_len_label_name = "len_label_name";
    INPUT_cls_logit_name = "cls_logit_name";
    INPUT_len_logit_name = "len_logit_name";
    OUTPUT_ocr_loss_name = "loss_name";
    MOD_osocr_loss_mod_name = "osocr_loss_mod_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.clsloss = this.register_mod(this.MOD_osocr_loss_mod_name, modcvt_dict);
        this.clslabel = this.register_input(this.INPUT_cls_label_name, iocvt_dict);
        this.lenlabel = this.register_input(this.INPUT_len_label_name, iocvt_dict);
        this.predcls = this.register_input(this.INPUT_cls_logit_name, iocvt_dict);
        this.predlen = this.register_input(this.INPUT_len_logit_name, iocvt_dict);
        this.lossname = this.register_output(this.OUTPUT_ocr_loss_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel)
        );
        workspace.objdict[this.lossname] = ocrloss;
        workspace.logdict[this.lossname] = terms;

        return workspace, environment;


class per_inst_ocr_loss_agent_mk2(ocr_loss_agent_mk2):
    INPUT_inst_mapping_name = "inst_mapping_name";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.cls_mapping = this.register_input(this.INPUT_inst_mapping_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel),
            workspace.get(this.cls_mapping)
        );
        workspace.add(this.lossname, per_inst_ocrloss);
        return workspace, environment;


def get_ocr_loss_agent_mk2(cls_label_name, cls_logit_name, len_label_name, len_logit_name,
                             ocr_loss_name,
                             osocr_loss_mod_name):
    engine = ocr_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}


def get_per_inst_ocr_loss_agent_mk2(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name,
                                      ):
    engine = per_inst_ocr_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}


if __name__ == '__main__':
    print(per_inst_ocr_loss_agent_mk2.get_default_configuration_scripts())
