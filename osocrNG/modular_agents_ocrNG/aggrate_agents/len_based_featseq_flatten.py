
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_idx_mapping
class neko_word_flatten_agent(neko_module_wrapping_agent):
    INPUT_feat_seq_name="feat_seq_name";
    INPUT_length_name="length_name";
    OUTPUT_flatten_feat_seq_name="flatten_feat_seq_name";
    OUTPUT_flatten_mapping_name="flatten_mapping";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat_seq=this.register_input(this.INPUT_feat_seq_name,iocvt_dict);
        this.len=this.register_input(this.INPUT_length_name,iocvt_dict);
        this.feat_seq_flatten=this.register_output(this.OUTPUT_flatten_feat_seq_name,iocvt_dict);
        this.mapping=this.register_output(this.OUTPUT_flatten_mapping_name,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        # if we don't know for sure how long it is, we guess.
        fout_emb,id  = neko_flatten_NG_idx_mapping.inflate(workspace.get(this.feat_seq), workspace.get(this.len));
        workspace.add(this.feat_seq_flatten,fout_emb);
        workspace.add(this.mapping,id);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                   feat_seq_name, length_name,
                   flatten_feat_seq_name, flatten_mapping_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_feat_seq_name: feat_seq_name, cls.INPUT_length_name: length_name,
                           cls.OUTPUT_flatten_feat_seq_name: flatten_feat_seq_name,
                           cls.OUTPUT_flatten_mapping_name: flatten_mapping_name}, "modcvt_dict": {}}}


# why not merging? bcs in 32x xtime can be computed on the fly, making a static xtime unnecessary .

class neko_word_flatten_agent_xtime(neko_word_flatten_agent):
    PARAM_xtime = "xtime";
    def set_etc(this, param):
        this.xtime = neko_get_arg(this.PARAM_xtime, param, 0); # do not expand time if not needed

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
            # if an a
        fout_emb,id  = neko_flatten_NG_idx_mapping.inflate(workspace.get(this.feat_seq), workspace.get(this.len)+this.xtime);
        workspace.add(this.feat_seq_flatten,fout_emb);
        workspace.add(this.mapping,id);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                   feat_seq_name, length_name,
                   flatten_feat_seq_name, flatten_mapping_name,
                   xtime
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_feat_seq_name: feat_seq_name, cls.INPUT_length_name: length_name,
                           cls.OUTPUT_flatten_feat_seq_name: flatten_feat_seq_name,
                           cls.OUTPUT_flatten_mapping_name: flatten_mapping_name}, cls.PARAM_xtime: xtime,
            "modcvt_dict": {}}}