from osocrNG.data_utils.common_data_presets_mk4.presets.names import P,VNIO,VN_PRFX,VN_STATE,VN_THING

# other variables.
class DATA_VN(VNIO):
    @classmethod
    def UNSAMPLED(this,prefix,what):
        return P(prefix,P("unsampled",what));

    @classmethod
    def TDICT_PATH(cls, prefix=""):
        return P(prefix, "tdict_file_path");
    @classmethod
    def TDICT_HASH(cls, prefix=""):
        return P(prefix, "tdict_file_hash");


    @classmethod
    def SIDE_INFO_UNSAMPLED(this,prefix=""):
        return  this.UNSAMPLED(prefix,this.SIDE_INFO());

    @classmethod
    def TDICT_UNSAMPLED(this,prefix=""):
        return this.UNSAMPLED(prefix,this.TDICT());
    @classmethod
    def UTF_UNSAMPLED(this,prefix=""):
        return this.UNSAMPLED(prefix,this.UTF());

    @classmethod
    def UTF_UNSAMPLED(this,prefix=""):
        return this.UNSAMPLED(prefix, this.UTF());

    @classmethod
    def DBG_SEL_UTF(this,prefix=""):
        return P(prefix,"selected_utf");
    @classmethod
    def DBG_MUX_UTF(this,prefix=""):
        return P(prefix,"muxed_utf");
    @classmethod
    def DBG_MUX_MSK(this,prefix=""):
        return P(prefix,"muxed_mask");
    @classmethod
    def DBG_MUX_ORI(this,prefix=""):
        return P(prefix,"muxed_orientation");
    @classmethod
    def DBG_SELECTED_FNT(this, prefix=""):
        return P(prefix, "debug_selected_utf");
    @classmethod
    def DBG_TFD(this, prefix=""):
        return P(prefix, "debug_token_fnt_dict");
    @classmethod
    def DBG_ID_DICT(this,prefix=""):
        return P(prefix,"id_dict");