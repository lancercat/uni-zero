import copy

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNKREP
from neko_sdk.log import warn

import torch
from torch.nn import functional as trnf

class ctc_tr_kit:
    @staticmethod
    def decode_ctc(maxp, tdict,blank_id):
        out_tok_full = [];
        out_tok=[];

        for i in range(0, maxp.shape[0]):
            prev="NEP_start_NEP";
            word_tok=[];
            word_text=[];
            for t in range(maxp.shape[1]):
                id=maxp[i,t];
                ch = tdict[id];
                word_tok.append(ch);
                if(id==blank_id):
                    prev="NEP_empty_NEP";
                else:
                    if(prev!=ch):
                        word_text.append(ch);
                        prev=ch;
            out_tok_full.append(word_tok);
            out_tok.append(word_text);
        return out_tok,out_tok_full;
    @staticmethod
    def decode_ctc_utf(maxp, utf_,blank_id):
        utf=copy.copy(utf_);
        utf.append(tUNKREP);
        out_tok_full = [];
        out_tok=[];
        for i in range(0, maxp.shape[0]):
            prev="NEP_start_NEP";
            word_tok=[];
            word_text=[];
            for t in range(maxp.shape[1]):
                id=maxp[i,t];
                ch = utf[id];
                word_tok.append(ch);
                if(id==blank_id):
                    prev="NEP_empty_NEP";
                else:
                    if(prev!=ch):
                        word_text.append(ch);
                        prev=ch;
            out_tok_full.append(word_tok);
            out_tok.append(word_text);
        return out_tok,out_tok_full;
class ctc_translate_agent_mk2_common(neko_module_wrapping_agent):
    INPUT_logit="logit";
    OUTPUT_pred_text="pred_text"
    OUTPUT_pred_token_list_name="pred_token_list_name";
    OUTPUT_full_token_list_name="full_token_list_name";
    PARAM_blank_id="blank_id";

    def set_etc(this, params):
        this.blank_id = neko_get_arg(this.PARAM_blank_id, params,0);
        assert (this.blank_id==0); # in mk2 we expect blank to be in tdict and holds id =0. Maybe in the future it will hold a setable  id...
        pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.logit = this.register_input(this.INPUT_logit, iocvt_dict);
        this.full_token_list_name = this.register_output(this.OUTPUT_full_token_list_name, iocvt_dict);
        this.pred_text = this.register_output(this.OUTPUT_pred_text, iocvt_dict);
        this.pred_token_list_name = this.register_output(this.OUTPUT_pred_token_list_name, iocvt_dict);
        pass;
    def get_decoded(this,workspace:neko_workspace,environment:neko_environment):
        outtok, out_full_tok, outpred=None,None,None;
        return outtok, out_full_tok,outpred;
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        outtok, out_full_tok, outpred = this.get_decoded(workspace, environment);
        try:
            outtok, out_full_tok, outpred=this.get_decoded(workspace,environment);
        except:
            nb=workspace.get(this.logit).shape[0];
            warn("badpred");
            # print(workspace.inter_dict);
            outtok = [["BADPRED"] for i in range(nb)];
            out_full_tok = [["BADPRED"] for i in range(nb)];
            outpred = ["BADPRED" for i in range(nb)];
        workspace.add(this.pred_text, outpred);
        workspace.add(this.full_token_list_name,out_full_tok);
        workspace.add(this.pred_token_list_name,outtok);
        return workspace, environment;

# it has no state(unlike loggers with histories), so it does not have a module.
class ctc_translate_agent_mk2(ctc_translate_agent_mk2_common):
    INPUT_tdict="tdict";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
    def get_decoded(this,workspace:neko_workspace,environment:neko_environment):
        logit = workspace.get(this.logit);
        tdict = workspace.get(this.tdict);
        with torch.no_grad():
            net_out = trnf.softmax(logit, dim=-1);
            maxp=torch.argmax(net_out,dim=-1).cpu().numpy();
        outtok, out_full_tok = ctc_tr_kit.decode_ctc(maxp, tdict,this.blank_id)
        outpred = ["".join(tok) for tok in outtok];
        return outtok, out_full_tok,outpred;


    @classmethod
    def get_agtcfg(cls,
                   logit, tdict,
                   full_token_list_name, pred_token_list_name,pred_text,
                   blank_id
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_logit: logit, cls.INPUT_tdict: tdict,
                                                        cls.OUTPUT_full_token_list_name: full_token_list_name,
                                                        cls.OUTPUT_pred_text: pred_text,
                                                        cls.OUTPUT_pred_token_list_name: pred_token_list_name},
                                         cls.PARAM_blank_id: blank_id, "modcvt_dict": {}}}


# it has no state(unlike loggers with histories), so it does not have a module.
class ctc_translate_agent_mk2_utf(ctc_translate_agent_mk2_common):
    INPUT_utf="utf";

    def get_decoded(this,workspace:neko_workspace,environment:neko_environment):
        logit = workspace.get(this.logit);
        utf = workspace.get(this.utf);
        with torch.no_grad():
            net_out = trnf.softmax(logit, dim=-1);
            maxp=torch.argmax(net_out,dim=-1).cpu().numpy();
        outtok, out_full_tok = ctc_tr_kit.decode_ctc_utf(maxp, utf,this.blank_id);
        outpred = ["".join(tok) for tok in outtok];
        return outtok, out_full_tok,outpred;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.utf = this.register_input(this.INPUT_utf, iocvt_dict);
        pass;



    @classmethod
    def get_agtcfg(cls,
                   logit, utf,
                   full_token_list_name, pred_token_list_name,pred_text,
                   blank_id
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_logit: logit, cls.INPUT_utf: utf,
                                                        cls.OUTPUT_full_token_list_name: full_token_list_name,
                                                        cls.OUTPUT_pred_text: pred_text,
                                                        cls.OUTPUT_pred_token_list_name: pred_token_list_name},
                                         cls.PARAM_blank_id: blank_id, "modcvt_dict": {}}}



if __name__ == '__main__':
    ctc_translate_agent_mk2.print_default_setup_scripts()