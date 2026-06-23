from neko_sdk.neko_framework_NG.libmeta.modules.side_info.abstract_mod import neko_abstract_meta_repo
from neko_sdk.neko_framework_NG.libmeta.file_names import FN


# don't put embeddings here!
# embedding will be fetched elswhere.
# we do this so this thing does not require gradients and can be put into another process.
class neko_id_side_info(neko_abstract_meta_repo):
    PARAM_root="root";
    PARAM_name="name";
    CONST_default_name=FN.ID_META;
    CONST_key_ids=FN.KEY_IDS;
    def __init__(this,param):
        super().__init__(param);
        if(this.CONST_key_ids not in this.meta):
            this.protos=[i for i in range(len(this.utf_list))];
        else:
            this.protos=this.meta[this.CONST_key_ids]; # if not
    def mk_proto(this):
        return this.protos;

