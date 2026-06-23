from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf
from neko_sdk.log import warn


import torch

class poswise_translate_agent_mk2(neko_module_wrapping_agent):
    INPUT_logit_name="logit_name";
    INPUT_tdict_name = "tdict_name";
    OUTPUT_pred_token_list_name="pred_token_list_name";
    OUTPUT_pred_string_name="pred_string_name";

    @classmethod
    def decode(cls,net_out, tdict):
        net_out = trnf.softmax(net_out, dim=-1);
        max_id=net_out.argmax(dim=-1).cpu().numpy();
        out =[];
        for cs in max_id:
            ss=[];
            for c in cs:
                if(c<0):
                    continue;
                if(c not in tdict):
                    warn("tdict proto mismatch");
                    continue;
                ss.append(tdict[c]);
            out.append(ss);

        # outpred = [[tdict[_]] if _ > 0 and _ <= len(tdict) else '' for _ in max_id];
        # for i in outpred:
        #     if (i == ""):
        #         pass;
        #     if (len(i) == 0):
        #         pass;
        return out

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.logit_name = this.register_input(this.INPUT_logit_name, iocvt_dict);
        this.tdict_name = this.register_input(this.INPUT_tdict_name, iocvt_dict);
        this.pred_string_name = this.register_output(this.OUTPUT_pred_string_name, iocvt_dict);
        this.pred_token_list_name = this.register_output(this.OUTPUT_pred_token_list_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        logit = workspace.get(this.logit_name);
        tdict = workspace.get(this.tdict_name);
        outpred=this.decode(
            logit,tdict
        );

        outstr=["".join(i) for i in outpred];

        workspace.add(this.pred_string_name,outstr);
        workspace.add(this.pred_token_list_name, outpred);
        pass;

    @classmethod
    def get_agtcfg(cls,
                   logit_name, tdict_name,
                   pred_string_name, pred_token_list_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": { cls.INPUT_logit_name: logit_name,
                           cls.INPUT_tdict_name: tdict_name, cls.OUTPUT_pred_string_name: pred_string_name,
                           cls.OUTPUT_pred_token_list_name: pred_token_list_name}, "modcvt_dict": {}}}
