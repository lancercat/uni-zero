from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

from torch.nn import functional as trnf

class neko_dense_xent_agent(ama):
    INPUT_logit="logit";
    INPUT_tlabel="tlabel";
    OUTPUT_dense_loss="dense_loss";
    PARAM_IGN_IDX="ign_idx";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.logit = this.register_input(this.INPUT_logit, iocvt_dict);
        this.tlabel = this.register_input(this.INPUT_tlabel, iocvt_dict);
        this.dense_loss = this.register_output(this.OUTPUT_dense_loss, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.IGN_IDX = neko_get_arg(this.PARAM_IGN_IDX, params,-1);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        logit = workspace.get(this.logit);
        tlabel = workspace.get(this.tlabel);
        workspace.add(this.dense_loss, trnf.cross_entropy(logit, tlabel, reduction="none",ignore_index=this.IGN_IDX ));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   logit, tlabel,
                   dense_loss,
                   IGN_IDX=-1
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_logit: logit, cls.INPUT_tlabel: tlabel, cls.OUTPUT_dense_loss: dense_loss},
            cls.PARAM_IGN_IDX: IGN_IDX, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_dense_xent_agent.print_default_setup_scripts()
