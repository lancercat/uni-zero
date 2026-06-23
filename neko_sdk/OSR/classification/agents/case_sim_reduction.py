from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_score_merging import scatter_cvt

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

import torch


# feat: *, #Part, #channel.
# center: * #part, #channel.
# *-> N,[T],
# return *, #classes
class neko_proto_sim_reduction_agent(neko_module_wrapping_agent):
    INPUT_proto_logit_name="proto_logits"; # backward_compatibility with 31x.
    INPUT_proto_label_name="plabel";
    OUTPUT_cls_logit_name="cls_logits";
    PARAM_dim="dim";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto_label_name = this.register_input(this.INPUT_proto_label_name, iocvt_dict);
        this.proto_logit_name = this.register_input(this.INPUT_proto_logit_name, iocvt_dict);
        this.cls_logit_name = this.register_output(this.OUTPUT_cls_logit_name, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.dim=neko_get_arg(this.PARAM_dim,params);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        proto_label_name = workspace.get(this.proto_label_name);
        proto_logit_name = workspace.get(this.proto_logit_name);
        cls_logits=scatter_cvt(proto_logit_name,proto_label_name,dim=this.dim);
        workspace.add(this.cls_logit_name,cls_logits);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   proto_label_name, proto_logit_name,
                   cls_logit_name,
                   dim=-1
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_proto_label_name: proto_label_name, cls.INPUT_proto_logit_name: proto_logit_name,
                           cls.OUTPUT_cls_logit_name: cls_logit_name}, "modcvt_dict": {},cls.PARAM_dim:dim}}



if __name__ == '__main__':
    neko_proto_sim_reduction_agent.print_default_setup_scripts()