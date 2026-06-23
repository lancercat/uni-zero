from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

import torch


# feat: *, #Part, #channel.
# center: * #part, #channel.
# *-> N,[T],
# return *, #classes
class simple_pred_agent(neko_module_wrapping_agent):
    MOD_pred_name="pred_name";
    INPUT_feat_name="feat_seq_name"; # backward_compatibility with 31x.
    INPUT_tensor_proto_vec_name="centers";
    INPUT_proto_label_name="plabel";
    OUTPUT_logit_name="logits";

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred = this.register(this.MOD_pred_name,modcvt_dict,this.mnames);

        this.feat_name=this.register(this.INPUT_feat_name,iocvt_dict,this.input_dict);
        this.tensor_proto_vec_name=this.register(this.INPUT_tensor_proto_vec_name,iocvt_dict,this.input_dict);

        this.proto_label_name=this.register(this.INPUT_proto_label_name,iocvt_dict,this.input_dict);
        this.logit_name=this.register(this.OUTPUT_logit_name,iocvt_dict,this.output_dict);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        workspace.inter_dict[this.logit_name] = environment.module_dict[this.pred](
            workspace.inter_dict[this.feat_name],
            workspace.inter_dict[this.tensor_proto_vec_name],
            workspace.inter_dict[this.proto_label_name]
        );
        return workspace;

    @classmethod
    def get_agtcfg(cls,
                   feat_name, proto_label_name, tensor_proto_vec_name,
                   logit_name,
                   pred_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_feat_name: feat_name, cls.INPUT_proto_label_name: proto_label_name,
                           cls.INPUT_tensor_proto_vec_name: tensor_proto_vec_name, cls.OUTPUT_logit_name: logit_name},
            "modcvt_dict": {cls.MOD_pred_name: pred_name}}}


if __name__ == '__main__':
    simple_pred_agent.print_default_setup_scripts()