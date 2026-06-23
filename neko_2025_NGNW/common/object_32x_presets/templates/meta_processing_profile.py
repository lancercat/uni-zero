from osocrNG.sptokens import tEOS,tBLANC

from neko_sdk.neko_framework_NG.names import P
class neko_metastream_processing_meta:
    MODALITY_TYPE_IM="image";
    MODALITY_TYPE_KEY = "key";
    MODALITY_TYPE_EMBEDDING = "embedding";
    MODALITY_TYPE_TOKSEQ = "token_sequence";

# just one modality. while it is possible that a meta may have several sideinfo branches,
# i see no need to jam stuff together

class neko_metastream_processing(neko_metastream_processing_meta):
    MODALITY="modality";
    CNTRED_SP_TOKENS="centered_sp_tokens";
    DATA_PRFX="data_prfx";
    MOD_PRFX="mod_prfx";
    CAPACITY="capacity";
    NAME="name";
    SIZE="size";
    MODSCP_ENDPOINTS="modscp_endpoints";
    FE_MOD_TIED="mod_tied"; # is it reusing modules elsewhere?
    # we don't have mixture of modality in 32x. maybe in the future we will.
    # also, we also assume metas are always precollated.
    @classmethod
    def get_default_im_meta_process(cls, data_data_prfx,capacity,size,data_mod_prfx,mod_tied=False):
        if(data_mod_prfx is None):
            data_mod_prfx=data_data_prfx;
        return {
            cls.CNTRED_SP_TOKENS:[tBLANC,tEOS],
            cls.DATA_PRFX:data_data_prfx,
            cls.MODALITY: cls.MODALITY_TYPE_IM,
            cls.MOD_PRFX:data_mod_prfx, # it just uses the dataprfx as default
            cls.FE_MOD_TIED: mod_tied,
            cls.NAME:data_data_prfx,
            cls.CAPACITY:capacity,
            cls.SIZE:size,
            cls.MODSCP_ENDPOINTS: [P(data_data_prfx,"armed")]
        };

