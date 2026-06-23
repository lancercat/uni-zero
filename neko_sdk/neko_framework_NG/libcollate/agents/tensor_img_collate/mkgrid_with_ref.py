from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
class neko_make_grid_with_ref(ama):
    INPUT_normed_imlist_ref = "normed_imlist_ref";
    INPUT_tensor_ref_map="tensor_ref_map";
    INPUT_tensor_ref_vec = "tensor_ref_vec";

    OUTPUT_tensor_grids="tensor_grids";
    MOD_collator="collator";
    pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.normed_imlist_ref = this.register_input(this.INPUT_normed_imlist_ref, iocvt_dict);

        this.tensor_ref_map = this.register_input(this.INPUT_tensor_ref_map, iocvt_dict);
        this.tensor_ref_vec = this.register_input(this.INPUT_tensor_ref_vec, iocvt_dict);

        this.grids = this.register_output(this.OUTPUT_tensor_grids, iocvt_dict);
        this.collator = this.register_mod(this.MOD_collator, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tensor_refrawim = workspace.get(this.normed_imlist_ref);

        if(workspace.has(this.tensor_ref_map)):
            tensor_refmap = workspace.get(this.tensor_ref_map);
        else:
            tensor_refmap=None;
        if (workspace.has(this.tensor_ref_vec)):
            tensor_refglob = workspace.get(this.tensor_ref_vec);
        else:
            tensor_refglob = None;

        workspace.add(
            this.grids,
            environment(this.collator,tensor_refrawim,tensor_refmap,tensor_refglob));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   raw_im,tensor_ref_fmap,tensor_ref_gfeat,
                   grids,
                   collator
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_normed_imlist_ref:raw_im,cls.INPUT_tensor_ref_map: tensor_ref_fmap,cls.INPUT_tensor_ref_vec:tensor_ref_gfeat,
                           cls.OUTPUT_tensor_grids: grids},
            "modcvt_dict": {cls.MOD_collator: collator}}}


