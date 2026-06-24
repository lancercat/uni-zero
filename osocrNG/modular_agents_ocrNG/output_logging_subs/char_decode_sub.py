from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.ocr_modules.io.encdec import decode_prob


class max_sim_length_pred(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.logits="logits";
        this.input_dict.length="length";
        this.input_dict.tdict="tdict";
        this.output_dict.pred_text="pred_text";
        pass;

    def fp(this,modular_dict, workspace):
        choutput, prdt_prob = decode_prob(workspace.inter_dict[this.input_dict.logits],
                                          workspace.inter_dict[this.input_dict.length],
                                          workspace.inter_dict[this.input_dict.tdict]);
        workspace.logdict[this.output_dict.pred_text]=choutput;
        return workspace;


