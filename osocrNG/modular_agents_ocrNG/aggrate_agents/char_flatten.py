import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_idx_mapping
# so why are we flatting chars? i dunno--- may be we will find a reason (multipart and compositional chars)?
class neko_single_roi_flatten_agent(neko_module_wrapping_agent):
    INPUT_feat_seq_name="feat_seq_name";
    OUTPUT_flatten_feat_seq_name="flatten_feat_seq_name";
    OUTPUT_flatten_mapping_name="flatten_mapping";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.feat_seq=this.register_input(this.INPUT_feat_seq_name,iocvt_dict);
        this.feat_seq_flatten=this.register_output(this.OUTPUT_flatten_feat_seq_name,iocvt_dict);
        this.mapping=this.register_output(this.OUTPUT_flatten_mapping_name,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fseq=workspace.get(this.feat_seq);
        # if we don't know for sure how long it is, we guess.
        fout_emb,id  = neko_flatten_NG_idx_mapping.inflate(fseq, torch.ones([fseq.shape[0]],dtype=torch.int,device=fseq.device));
        workspace.add(this.feat_seq_flatten,fout_emb);
        workspace.add(this.mapping,id);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                   feat_seq_name,
                   flatten_feat_seq_name, flatten_mapping_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_feat_seq_name: feat_seq_name,
                           cls.OUTPUT_flatten_feat_seq_name: flatten_feat_seq_name,
                           cls.OUTPUT_flatten_mapping_name: flatten_mapping_name}, "modcvt_dict": {}}};


