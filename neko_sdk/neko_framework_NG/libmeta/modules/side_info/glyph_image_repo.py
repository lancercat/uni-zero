from neko_sdk.neko_framework_NG.libmeta.modules.side_info.abstract_mod import neko_abstract_meta_repo
from neko_sdk.neko_framework_NG.libmeta.file_names import FN



class neko_embedding_proto_normalized(neko_abstract_meta_repo):
    PARAM_root="root";
    PARAM_name="name";
    CONST_default_name=FN.GLYPH_META;
    def __init__(this,param):
        super().__init__(param);
        this.protos = [this.meta[i] for i in this.utf_list];

    def mk_proto(this):
        return this.protos
