from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf
from neko_sdk.log import warn


import torch

class lpos_translate_agent_mk2(neko_module_wrapping_agent):
    INPUT_logit_name="logit_name";
    INPUT_tdict_name = "tdict_name";
    INPUT_length_name = "length_name";
    OUTPUT_pred_token_list_name="pred_token_list_name";
    OUTPUT_pred_string_name="pred_string_name";

    @classmethod
    def decode_prob(cls,net_out, length, tdict):
        out = []
        out_prob = []
        net_out = trnf.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:,
                               0].tolist()
            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]

            current_text = [tdict[_] if _ > 0 and _ <= len(tdict) else '' for _ in current_idx_list]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_prob)

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.length_name = this.register_input(this.INPUT_length_name, iocvt_dict);
        this.logit_name = this.register_input(this.INPUT_logit_name, iocvt_dict);
        this.tdict_name = this.register_input(this.INPUT_tdict_name, iocvt_dict);
        this.pred_string_name = this.register_output(this.OUTPUT_pred_string_name, iocvt_dict);
        this.pred_token_list_name = this.register_output(this.OUTPUT_pred_token_list_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        length = workspace.get(this.length_name);
        logit = workspace.get(this.logit_name);
        tdict = workspace.get(this.tdict_name);
        outpred = this.decode_prob(
            logit, length, tdict
        )[0];
        outstr = ["".join(i) for i in outpred];
        try:
            outpred=this.decode_prob(
                logit,length,tdict
            )[0];
            outstr=["".join(i) for i in outpred];
        except:
            warn("badpred");
            # print(workspace.inter_dict);
            outpred=[["BADPRED"] for i in range(length.shape[0])];
            outstr = ["BADPRED" for i in range(length.shape[0])];

        workspace.add(this.pred_string_name,outstr);
        workspace.add(this.pred_token_list_name, outpred);

        pass;

    @classmethod
    def get_agtcfg(cls,
                   length_name, logit_name, tdict_name,
                   pred_string_name, pred_token_list_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_length_name: length_name, cls.INPUT_logit_name: logit_name,
                           cls.INPUT_tdict_name: tdict_name, cls.OUTPUT_pred_string_name: pred_string_name,
                           cls.OUTPUT_pred_token_list_name: pred_token_list_name}, "modcvt_dict": {}}}
