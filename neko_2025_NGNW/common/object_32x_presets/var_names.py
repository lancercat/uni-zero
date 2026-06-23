from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN,VN_THING,VN_STATE,VN_PRFX


from neko_sdk.neko_framework_NG.names import P

class project_32x_varnames(DATA_VN):
    @classmethod
    def HISTY(cls,prefix):
        return VN_PRFX.SAM(prefix, "routing_history");
    @classmethod
    def LOG_LIKLYHD(cls,prefix):
        return VN_PRFX.SAM(prefix, "presence_log_likelihood");

    @classmethod
    def DSCRGED(cls, prefix):
        return VN_PRFX.SAM(prefix, "discouraged_state");
    @classmethod
    def ACT_LOGIT(cls,prefix):
        return VN_PRFX.SAM(prefix,"action_logit");
    @classmethod
    def ACT_DSCRGED_MSK(cls,prefix):
        return VN_PRFX.SAM(prefix,P("is_discouraged_action","mask"));
    @classmethod
    def ACTS_SAMED_TEN(cls, prefix):
        return VN_PRFX.SAM(prefix, P("actions_sampled","tensor"));
    @classmethod
    def ACTS_SAMED_LST(cls, prefix):
        return VN_PRFX.SAM(prefix, "actions_sampled");
    @classmethod
    def ACT_DSCRGED_LST(cls,prefix):
        return VN_PRFX.SAM(prefix,P("is_discouraged_action","list"));
    @classmethod
    def ACT_LOG_PROBS(cls, prefix):
        return VN_PRFX.SAM(prefix, "actions_log_probability");

    @classmethod
    def ACT_NAMES(cls, prefix):
        return VN_PRFX.SAM(prefix, "action_names"); # well names of actions, for logging purpose

    @classmethod
    def ACT_FRBDN(cls,prefix):
        return VN_PRFX.SAM(prefix,"forbidden_action_mask");
    @classmethod
    def ACT_SAMK(cls,prefix):
        return VN_PRFX.SAM(prefix,"sample_top_k_override");

    @classmethod
    def ATT_MAP_GA(cls, prefix=""):
        return P(prefix, "attention_map_geometric");


    @classmethod
    def WORD_FEAT_MAP_LIST_DETACHED(cls,prefix=""):
        return P(cls.FEAT_MAP_LST(prefix),"detached");
    @classmethod
    def WORD_FEAT_MAP_LIST_DETACHED_SE(cls, prefix=""):
        return P(cls.WORD_FEAT_MAP_LIST_DETACHED(prefix), "se");

    # if there is a need for temporal features to be extracted.
    @classmethod
    def WORD_TEMP_FEAT_MAP_LAST(cls,prefix=""):
        return P(prefix,"temporal_feature_map_last");
    @classmethod
    def WORD_TEMP_FEAT_MAP_LIST(cls,prefix=""):
        return P(prefix,"temporal_feature_map_list");

    @classmethod
    def WORD_TEMP_FEAT_GPOOL(cls,prefix=""):
        return P(prefix,"temporal_feature_gpool");

    @classmethod
    def WORD_TEMP_FEAT_GLOBAL(cls,prefix=""):
        return P(prefix,"temporal_feature_global");
    @classmethod
    def WORD_LEN_PRED_LOGIT(cls,prefix=""):
        return P(prefix,"length_prediction_logit");
    @classmethod
    def WORD_LEN_PRED_VALUE(cls,prefix=""):
        return P(prefix,"length_prediction_value");
    @classmethod
    def DBG_ATT_IMS(cls,prefix=""):
        return P(prefix,"debug_attention_images");
    @classmethod
    def WORD_LEN_GT_VALUE(cls, prefix=""):
        return VN_PRFX.GT_TAINT(prefix, "length_value");

    @classmethod
    def WORD_ATT_MAP(cls, prefix=""):
        return P(prefix, "attention_map_temporal");
    @classmethod
    def WORD_FEAT_FLATTENED(cls,prefix=""):
        return P(prefix,"word_feat_flattened"); # i know there will be duplications, but let's dedup later.

    @classmethod
    def ROI_FEAT_SEQ(cls,prefix=""):
        return P(prefix, "roi_feat_sequential");


    @classmethod
    def FLATTEN_ROI_FEAT_SEQ(cls,prefix):
        return P(prefix, cls.ROI_FEAT_SEQ("flatten")); # not always needed_ unless you are using danish or abish


    @classmethod
    def FLATTEN_ROI_LOGIT_SEQ(cls,prefix=""):
        return P(prefix, "flatten_roi_cls_logit_seq");
    @classmethod
    def FLATTEN_WORD_GT_SEQ(cls,prefix=""):
        return P(prefix, "flatten_word_groundtruth_seq");
    @classmethod
    def FLATTEN_MAP(cls, prefix):
        return P(prefix, "flatten_map");  # not always needed_ unless you are using danish or abish


    @classmethod
    def RAW_THUMBNAIL(cls, prefix=""):
        return P(prefix, "numpy_thumbnail");

    @classmethod
    def LEN_LOSS_PER_INST(this,prefix=""):
        return P(prefix,"len_loss_per_inst");
    @classmethod
    def CLS_LOSS_PER_INST(this,prefix=""):
        return P(prefix,"cls_loss_per_inst");

    @classmethod
    def DENCE_CLS_LOSS(this, prefix=""):
        return P(prefix, "dense_cls_loss");

    @classmethod
    def NED_PER_INST(this,prefix=""):
        return P(prefix,"normed_edit_distance_per_inst");


    @classmethod
    def TENSOR_THUMBNAIL_ATTMAP(cls, prefix=""):
        return P(prefix, "sample_thumbnail_attention");

    # in 32x, feature twr and last feature maps are separated.
    @classmethod
    def THUMB_FEATURE_TWR(cls,prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_tower");
    # the feature tower now comes with tags for each feature map, like "2x, 4x,.... "
    @classmethod
    def THUMB_FEATURE_TWR_tags(cls, prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_tags");

    @classmethod
    def THUMB_FEATUREMAP(cls,prefix=""):
        return P(prefix, "sample_thumbnail_featuremap");
    @classmethod
    def THUMB_FEATUREMAPSE(cls,prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_se");

    @classmethod
    def THUMB_FEATUREMAP_TWR(cls, prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_tower");
    @classmethod
    def THUMB_FEATURETWR_TAG(cls, prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_tower_tags");

    @classmethod
    def THUMB_FEATUREMAP_SE(cls, prefix=""):
        return P(prefix, "sample_thumbnail_featuremap_se");

    @classmethod
    def THUMB_FEATUREMAP(cls, prefix=""):
        return P(prefix, "sample_thumbnail_featuremap");

    @classmethod
    def THUMB_FEATURE(cls,prefix=""):
        return P(prefix, "sample_thumbnail_feature_vector");
    pass;

    @classmethod
    def COSSIM(cls, prefix=""):
        return P(prefix, "cosine_similarity");

    @classmethod
    def COSSIM_WUNK(cls, prefix=""):
        return P(prefix, "cosine_similarity_with_unknown");

    @classmethod
    def SIMCONF(cls, prefix=""):
        return P(prefix, "similarity_confidence");
    pass;




