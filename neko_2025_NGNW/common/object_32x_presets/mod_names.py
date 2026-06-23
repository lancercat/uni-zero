from neko_sdk.neko_framework_NG.names import P


class project_32x_modnames:

    @classmethod
    def CLS_EMB(cls,prefix=""):
        return P(prefix,"classifier_centers");
    @classmethod
    def GLOBAL_MVN(cls,prefix=""):
        return P(prefix,"global_mvn");
    @classmethod
    def ACT_SEL(cls,prefix=""):
        return P(prefix,"action_selector");
    @classmethod
    def ACT_SPL(cls,prefix=""):
        return P(prefix,"action_sampler");

    @classmethod
    def SAM_COLLATE(cls,prefix=""):
        return P(prefix,"sam_collate");
    @classmethod
    def FE(cls,prefix=""):
        return P(prefix,"fe");
    @classmethod
    def SE(cls,prefix=""):
        return P(prefix, "se");
    @classmethod
    def ATTN(cls,prefix=""):
        return P(prefix, "attn");
    @classmethod
    def WORD_LEN_PRED(cls,prefix=""):
        return P(prefix, "lpred");


    @classmethod
    def FE_conv(cls, prefix=""):
        return P(prefix, "convs");

    @classmethod
    def FE_ffns(cls, prefix=""):
        return P(prefix, "ffns");
    @classmethod
    def FE_bn(cls,prefix=""):
        return P(prefix,"bns");

    @classmethod
    def WORD_TEMPORAL_SE(cls, prefix):
        return P(prefix, "word_tse");
    @classmethod
    def WORD_TEMPORAL_FE(cls, prefix):
        return P(prefix, "word_tfe");
    @classmethod
    def WORD_TEMPORAL_GFE(cls, prefix):
        return P(prefix, "word_tfe_digest");
    @classmethod
    def WORD_ATTN(cls, prefix=""):
        return P(prefix, "word_attn");
    @classmethod
    def INST_SEG_MSK_HEAD(cls, prefix=""):
        return P(prefix, "insseg_mask_head");
    @classmethod
    def WORD_LPRED(cls, prefix=""):
        return P(prefix, "word_lenpred");

    @classmethod
    def WORD_AGGR(cls, prefix=""):
        return P(prefix, "word_aggr");

    @classmethod
    def META_SPHDR(cls, prefix=""):
        return P(prefix, "special_token_prototypes_holader");
    @classmethod
    def META_EMB(cls, prefix=""):
        return P(prefix, "embeddings");

    @classmethod
    def META_SPINJ(cls, prefix=""):
        return P(prefix, "special_token_prototypes_injecter");

    # in 32x classifier is tied to meta.
    @classmethod
    def META_CLASSIFIER(cls, prefix=""):
        return P(prefix, "classifer");
    @classmethod
    def META_TOKENIZER(cls, prefix=""):
        return P(prefix, "tokenizer");


    @classmethod
    def CATT(cls,prefix=""):
        return P(prefix,"char_attention");
    @classmethod
    def ATTAGG(cls,prefix=""):
        return P(prefix,"attention_aggregation");
    @classmethod
    def CLASSIFIER(cls, prefix=""):
        return P(prefix, "classifer");


    @classmethod
    def PER_INSTANCE_OCR_LPRED_LOSS(cls,prefix=""):
        return P(prefix, "per_inst_length_loss");
    def PER_INSTANCE_OCR_CLS_LOSS_NAME(cls,prefix=""):
        return P(prefix, "per_inst_cls_loss");
    @classmethod
    def UNK_TH(cls,prefix):
        return P(prefix, "unk_threshold");
    @classmethod
    def SIM(cls,prefix):
        return P(prefix, "similarity");
    # @classmethod
    # def UNK_THRESH(cls, prefix=""):
    #     return P(prefix, "unk_thresh");


