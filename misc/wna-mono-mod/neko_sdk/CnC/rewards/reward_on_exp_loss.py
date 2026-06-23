import torch.nn
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# We will make parameter free module agents, to reduce coding complexity.


# since it writes to loss dict, so it stays a specific module
class nekolog_exp_penalty(neko_module_wrapping_agent):
    INPUT_allsamp="all_samp";
    INPUT_per_action_penalty = "per_action_penalty";
    INPUT_mlogp = "minus_log_probability";
    OUTPUT_weightenalty = "penalty_on_loss";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.loss = this.register_input(this.INPUT_per_action_penalty, iocvt_dict);
        this.mlogp = this.register_input(this.INPUT_mlogp, iocvt_dict);
        this.samples=this.register_input(this.INPUT_allsamp,iocvt_dict);

        this.wpenalty = this.register_output(this.OUTPUT_weightenalty, iocvt_dict);

    # Think: What is the baseline here? Do we have one? Do we need one?
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        mlogp=workspace.get(this.mlogp);
        sams=workspace.get(this.samples);

        penalty = torch.tensor(workspace.get(this.loss)).detach()*torch.stack(mlogp);
        penalty=penalty.sum()/len(sams); # actions are sampled on the number of total samples
        workspace.add_loss(this.wpenalty, penalty);
        workspace.logdict[this.wpenalty]=penalty.item();
        return workspace, environment;


def get_nekolog_exp_penalty(sample_name,mlogp_name, per_action_penalty_name,
                                        penalty_name,
                                        ):
    engine = nekolog_exp_penalty;
    return {
        "agent": engine,
        "params": {"iocvt_dict": {engine.INPUT_allsamp:sample_name,engine.INPUT_mlogp: mlogp_name,
                                 engine.INPUT_per_action_penalty: per_action_penalty_name,
                                 engine.OUTPUT_weightenalty: penalty_name}, "modcvt_dict": {}}}


if __name__ == '__main__':
    print(nekolog_exp_penalty.get_default_configuration_scripts())
