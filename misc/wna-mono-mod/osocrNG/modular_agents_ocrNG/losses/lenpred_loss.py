from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment



class ocr_loss_agent_mk2(neko_module_wrapping_agent):
    INPUT_len_label_name = "len_label_name";
    INPUT_len_logit_name = "len_logit_name";
    OUTPUT_len_loss_name = "loss_name";
    MOD_osocr_length_loss_mod_name = "osocr_length_loss_mod_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.lenloss = this.register_mod(this.MOD_osocr_length_loss_mod_name, modcvt_dict);
        this.lenlabel = this.register_input(this.INPUT_len_label_name, iocvt_dict);
        this.predlen = this.register_input(this.INPUT_len_logit_name, iocvt_dict);
        this.lossname = this.register_output(this.OUTPUT_len_loss_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        loss, terms = environment.module_dict[this.lenloss](
             workspace.get(this.predlen),
             workspace.get(this.lenlabel)
        );
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname] = terms;

        return workspace, environment;


class per_inst_len_loss_agent_mk2(ocr_loss_agent_mk2):

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        loss= environment.module_dict[this.lenloss](
             workspace.get(this.predlen),
             workspace.get(this.lenlabel)
        );
        workspace.add(this.lossname, loss);
        return workspace, environment;
def get_ocr_loss_agent_mk2(len_label_name,len_logit_name, len_loss_name,
                           osocr_length_loss_mod_name):
    engine = ocr_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_len_label_name: len_label_name, engine.INPUT_len_logit_name: len_logit_name, engine.OUTPUT_len_loss_name: len_loss_name}, "modcvt_dict": {engine.MOD_osocr_length_loss_mod_name: osocr_length_loss_mod_name}}}

def get_per_inst_len_loss_agent_mk2(len_label_name,len_logit_name,len_loss_name,osocr_length_loss_mod_name):
    engine = per_inst_len_loss_agent_mk2;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_len_label_name: len_label_name, engine.INPUT_len_logit_name: len_logit_name, engine.OUTPUT_len_loss_name: len_loss_name}, "modcvt_dict": {engine.MOD_osocr_length_loss_mod_name: osocr_length_loss_mod_name}}}

if __name__ == '__main__':
    print(per_inst_len_loss_agent_mk2.get_default_configuration_scripts())
