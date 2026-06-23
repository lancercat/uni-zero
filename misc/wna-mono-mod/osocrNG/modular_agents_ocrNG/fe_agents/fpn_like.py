from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment;

class fpn_like_fe(neko_module_wrapping_agent):
    INPUT_feature_maps="feats";
    OUTPUT_last_feat="last_feat";
    OUTPUT_endpoints="endpoints"
    MOD_fpn="fpn_mod"
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input=this.register_input_list(this.INPUT_feature_maps,iocvt_dict);
        this.endpoints=this.register_output_list(this.OUTPUT_endpoints, iocvt_dict);
        this.output=this.register_output(this.OUTPUT_last_feat,iocvt_dict);
        this.fpnmod=this.register_mod(this.MOD_fpn,modcvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        l,e=environment(this.fpnmod,workspace.get(this.input));
        workspace.add(this.output,l);
        workspace.add(this.endpoints,e);
        return workspace,environment;
def get_fpn_like_fe(feature_maps_name,endpoints_name,last_feat_name,fpn_name):
    engine = fpn_like_fe;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_feature_maps: feature_maps_name, engine.OUTPUT_endpoints: endpoints_name, engine.OUTPUT_last_feat: last_feat_name}, "modcvt_dict": {engine.MOD_fpn: fpn_name}}}

if __name__ == '__main__':
    print(fpn_like_fe.get_default_configuration_scripts());