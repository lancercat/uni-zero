from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama

class neko_multipart_sim_sample_proto(ama):
    INPUT_sample_feat="sample_feat";
    INPUT_protos="protos";
    OUTPUT_cossim="cossim";
    OUTPUT_confidence="confidence";
    MOD_sim_mod="sim_mod";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.protos = this.register_input(this.INPUT_protos, iocvt_dict);
        this.sample_feat = this.register_input(this.INPUT_sample_feat, iocvt_dict);
        this.confidence = this.register_output(this.OUTPUT_confidence, iocvt_dict);
        this.cossim = this.register_output(this.OUTPUT_cossim, iocvt_dict);
        this.sim_mod = this.register_mod(this.MOD_sim_mod, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        protos = workspace.get(this.protos);
        sample_feat = workspace.get(this.sample_feat);
        cos,conf=environment(this.sim_mod,sample_feat,protos);
        workspace.add(this.cossim,cos);
        workspace.add(this.confidence,conf);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                    sample_feat,protos,
                    cossim,confidence,
                   sim_mod
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_protos: protos, cls.INPUT_sample_feat: sample_feat,
                                                        cls.OUTPUT_confidence: confidence, cls.OUTPUT_cossim: cossim},
                                         "modcvt_dict": {cls.MOD_sim_mod: sim_mod}}}

if __name__ == '__main__':
    neko_multipart_sim_sample_proto.print_default_setup_scripts()