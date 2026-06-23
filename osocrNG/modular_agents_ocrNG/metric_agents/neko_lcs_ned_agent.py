from osocrNG.utils.ed_lcs import edit_distance, longest_common_subsequence
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_ed_agent(ama):
    INPUT_pred="prediction";
    INPUT_gt="gt";
    OUTPUT_normed_ed="normed_ed";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.gt = this.register_input(this.INPUT_gt, iocvt_dict);
        this.pred = this.register_input(this.INPUT_pred, iocvt_dict);
        this.normed_ed = this.register_output(this.OUTPUT_normed_ed, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt = workspace.get(this.gt);
        pred = workspace.get(this.pred);
        neds=[];
        for g,p in zip(gt,pred):
            levenshtein_distance=edit_distance(g,p);
            ned=levenshtein_distance/max(len(g),1); # it is normalized by gt len, as otherwise model may do the overlength hack
            neds.append(ned);
        workspace.add(this.normed_ed,neds)
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   gt, pred,
                    normed_ed
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_gt: gt, cls.INPUT_pred: pred,
                           cls.OUTPUT_normed_ed: normed_ed}, "modcvt_dict": {}}}

class neko_lcs_agent(ama):
    INPUT_pred="prediction";
    INPUT_gt="gt";
    OUTPUT_lcs_idx_gt="lcs_gt";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.gt = this.register_input(this.INPUT_gt, iocvt_dict);
        this.pred = this.register_input(this.INPUT_pred, iocvt_dict);
        this.lcs_idx_gt = this.register_output(this.OUTPUT_lcs_idx_gt, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        gt = workspace.get(this.gt);
        pred = workspace.get(this.pred);
        lcss=[];
        for g,p in zip(gt,pred):
            lcs_indices_gt=longest_common_subsequence(g,p);
            lcss.append(lcs_indices_gt);
        workspace.add(this.lcs_idx_gt,lcss);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   gt, pred,
                   lcs_idx_gt
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_gt: gt, cls.INPUT_pred: pred, cls.OUTPUT_lcs_idx_gt: lcs_idx_gt}, "modcvt_dict": {}}}
