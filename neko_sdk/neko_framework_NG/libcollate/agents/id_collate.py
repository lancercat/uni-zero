import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.cfgtool.argsparse import neko_get_arg
from multiprocess.pool import Pool

# when sideinfo is just some global id--- that werks as well...
# they remain global ids after sampling. (tdict will be local ids after sampling)
# you can use them to fetch embeddings if you so wish...
# though a better (though more complex) solution is to build it on the fly with proto-utf.
# we will have knowledge modules later to accomodate the utf part--- use id only if you have to
class neko_id_collate(ama):
    INPUT_raw_ids="raw_images";
    OUTPUT_tensor_ids="tensor_images";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.raw_ids = this.register_input(this.INPUT_raw_ids, iocvt_dict);
        this.tensor_ids = this.register_input(this.OUTPUT_tensor_ids, iocvt_dict);
        pass;


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_ids = workspace.get(this.raw_ids);
        workspace.add(this.tensor_ids,torch.tensor(raw_ids,dtype=torch.int32));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   raw_ids,
                   tensor_ids
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_raw_ids: raw_ids, cls.OUTPUT_tensor_ids: tensor_ids},
                                         "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_id_collate.print_default_setup_scripts()
