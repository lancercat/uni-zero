from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_squeeze_dims_agent(ama):
    INPUT_feat="in";
    OUTPUT_squeezed_feat="out";
    PARAM_dims_to_squeeze="dims";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat = this.register_input(this.INPUT_feat, iocvt_dict);
        this.squeezed_feat = this.register_output(this.OUTPUT_squeezed_feat, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.dims_to_squeeze = neko_get_arg(this.PARAM_dims_to_squeeze, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        feat = workspace.get(this.feat);
        workspace.add(this.squeezed_feat,feat.squeeze(this.dims_to_squeeze));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   feat,
                   squeezed_feat,
                   dims_to_squeeze
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_feat: feat, cls.OUTPUT_squeezed_feat: squeezed_feat},
                                         cls.PARAM_dims_to_squeeze: dims_to_squeeze, "modcvt_dict": {}}}

class neko_unsqueeze_dims_agent(ama):
    INPUT_feat="in";
    OUTPUT_unsqueezed_feat="out";
    PARAM_dims_to_squeeze="dims";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat = this.register_input(this.INPUT_feat, iocvt_dict);
        this.unsqueezed_feat = this.register_output(this.OUTPUT_unsqueezed_feat, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.dims_to_squeeze = neko_get_arg(this.PARAM_dims_to_squeeze, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        feat = workspace.get(this.feat);
        workspace.add(this.unsqueezed_feat,feat.unsqueeze(this.dims_to_squeeze));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   feat,
                   unsqueezed_feat,
                   dims_to_squeeze
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_feat: feat, cls.OUTPUT_unsqueezed_feat: unsqueezed_feat},
                                         cls.PARAM_dims_to_squeeze: dims_to_squeeze, "modcvt_dict": {}}}



if __name__ == '__main__':
    neko_squeeze_dims_agent.print_default_setup_scripts()