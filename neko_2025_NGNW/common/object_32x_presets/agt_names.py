from neko_sdk.neko_framework_NG.names import P

class project_32x_agtnames:
    @classmethod
    def ROUTER_INIT(cls,prefix):
        return P(prefix,"router_init");
    @classmethod
    def VISPROTO_CACHE(cls,prefix):
        return P(prefix,"vision_prototype_cache");
    @classmethod
    def DCLS(cls,prefix):
        return P(prefix,"dense_classification")
    @classmethod
    def ROUTER_MKACT(cls, prefix):
        return P(prefix, "make_action");
    @classmethod
    def ROUTER_ROUTE(cls, prefix):
        return P(prefix, "route");
    @classmethod
    def RES_AGGR(cls,prefix):
        return P(prefix, "result_aggregation");
    @classmethod
    def MKGRID(cls,prefix):
        return P(prefix, "make_grid");

    @classmethod
    def GRIDCOLL(cls, prefix):
        return P(prefix, "grid_collation");

    @classmethod
    def SAMPL(cls,prefix):
        return P(prefix,"sampling");
    @classmethod
    def C2FKSEQ(cls,prefix):
        return P(prefix,"char_to_fake_sequence");
    @classmethod
    def FE(cls,prefix=""):
        return P(prefix,"fe");
    @classmethod
    def TFE(cls,prefix=""):
        return P(prefix, "tfe");
    @classmethod
    def HEAD_SEL(cls,prefix):
        return P(prefix,"head_sel");

    @classmethod
    def HEAD_SERIALIZE(cls,prefix):
        return P(prefix,"head_serialize");

    @classmethod
    def TODEV(cls, prefix):
        return P(prefix, "to_device");

    @classmethod
    def ASS(cls, prefix=""):
        return P(prefix, "asmt");

    @classmethod
    def MKPROTO(cls, prefix=""):
        return P(prefix, "mk_proto");

    @classmethod
    def SP_INJECTION(cls,prefix=""):
        return P(prefix,"fe_injection");
    # this will tell you the agent is training only.
    # for var names, it is more complex

    @classmethod
    def TRAIN(cls,what,prefix):
        return P(prefix,P("train",what));
    @classmethod
    def INST_MSK(cls,prefix):
        return P(prefix,"instance_mask");
    @classmethod
    def ATT_AGGR(cls,prefix):
        return P(prefix,"att_aggr");
    @classmethod
    def POL_LRN(cls,prefix):
        return P(prefix,"policy_learning");

    @classmethod
    def fe_agent(cls,prefix=""):
        return P(prefix,"feature_extraction");
    @classmethod
    def meta_decoration_agent(cls,prefix=""):
        return P(prefix,"meta_decoration");
    @classmethod
    def PRED(cls,prefix):
        return P(prefix,"prediction");

    @classmethod
    def TRANS(cls,prefix):
        return P(prefix,"translation");
    @classmethod
    def PRED_LO(cls, prefix):
        return P(prefix, "prediction_low");
    @classmethod
    def PRED_LO_TR(cls, prefix):
        return P(prefix, "prediction_low_training");

    @classmethod
    def MKGT(cls, prefix):
        return P(prefix, "gt_making");

    @classmethod
    def flatten_agent(cls,prefix):
        return P(prefix,"prediction");
    # during testing, the centers will be cacheable
    @classmethod
    def TE_side_info_to_centers(cls, prefix=""):
        return cls.TE_("make_center",prefix);

