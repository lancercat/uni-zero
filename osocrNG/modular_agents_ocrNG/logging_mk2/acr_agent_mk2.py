
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.modular_agents_ocrNG.logging_mk2.reporter_ACR_mk2 import reporter_core_ACR_mk2
import sys

class acr_fps_reporter_mk2(neko_module_wrapping_agent):
    INPUT_pred_text_tokens_name="in_pred_text_tokens";
    INPUT_raw_label_tokens_name="in_raw_label_tokens";
    INPUT_tdict_name="in_tdict";
    OUTPUT_results_name="results_name";
    PARAM_protocol_force_disable_unk="disable_unk";
    PARAM_name="name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.pred_text_tokens_name = this.register_input(this.INPUT_pred_text_tokens_name, iocvt_dict);
        this.raw_label_tokens_name = this.register_input(this.INPUT_raw_label_tokens_name, iocvt_dict);
        this.tdict_name = this.register_input(this.INPUT_tdict_name, iocvt_dict);
        this.results_name = this.register_output(this.OUTPUT_results_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.name = neko_get_arg(this.PARAM_name, params);
        this.disable_unk = neko_get_arg(this.PARAM_protocol_force_disable_unk, params);

        pass;

    def reset(this):
        this.recorder=reporter_core_ACR_mk2(this.name,this.disable_unk);
        # due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(this.tdict_name in workspace.inter_dict):
            tdict=workspace.inter_dict[this.tdict_name];
        else:
            tdict=None;
        if(this.raw_label_tokens_name not in workspace.inter_dict):
            return; # no gt, no report
        this.recorder.record_batch(
            workspace.get(this.raw_label_tokens_name),
            workspace.get(this.pred_text_tokens_name),tdict);


    def report(this,workspace:neko_workspace,environment:neko_environment):
        rpt=this.recorder.report(environment.epoch_idx,environment.batch_idx);
        print(" ,".join(k+": "+str(rpt[k]) for k in rpt));
        sys.stdout.flush();
        workspace.add_log(this.results_name,rpt);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                   pred_text_tokens_name, raw_label_tokens_name, tdict_name,
                   results_name,
                   name,disable_unk
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_pred_text_tokens_name: pred_text_tokens_name,
                                                        cls.INPUT_raw_label_tokens_name: raw_label_tokens_name,
                                                        cls.INPUT_tdict_name: tdict_name,
                                                        cls.OUTPUT_results_name: results_name}, cls.PARAM_name: name,cls.PARAM_protocol_force_disable_unk:disable_unk,
                                         "modcvt_dict": {}}}

if __name__ == '__main__':
    acr_fps_reporter_mk2.print_default_setup_scripts();
