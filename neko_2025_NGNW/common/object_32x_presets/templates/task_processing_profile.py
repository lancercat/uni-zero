from neko_sdk.neko_framework_NG.names import P
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN

class neko_task_processing_meta:
    TASK_TYPE_LENPREDCLS="lenpred_cls";
    TASK_TYPE_SINGLE_CLS = "single_classification";
    TASK_TYPE_XTCS="cross_task_cosine_similarty";
    TASK_TYPE_SEQ="sequence";
    TASK_TYPE_SEMSEG="semseg";
    TASK_TYPE_INSSEG_CA="insseg_class_agnostic";


# just one modality. while it is possible that a meta may have several sideinfo branches,
# i see no need to jam stuff together
class neko_seq_task_param:
    TASK_SEQ_MAXT= "maxT";
    TASK_SEQ_ORDERED= "ordered"; # is the output ordered?

    TASK_SEQ_IMPL_SEQ_PRES="detr";
    TASK_SEQ_IMPL_SEQ_LPOS = "lpos";
    TASK_SEQ_IMPL_CTC="ctc";
    TASK_SEQ_IMPL_SEQ_EOS="seq_eos";
class neko_xcts_task_param:
    TASK_XCTS_GTDICTS="global_tdicts";
    TASK_XCTS_SEQ_UTF_EPS="xcts_utf_sequence_eps";
    TASK_XCTS_SEQ_FEAT_EPS = "xcts_feat_sequence_eps";

    TASK_XCTS_ITEM_UTF_EPS = "xcts_utf_item_eps";
    TASK_XCTS_ITEM_FEAT_EPS = "xcts_feat_item_eps";

    # if there are gts then detach them and align what not to gts.
    TASK_XCTS_GT_ITEM_UTF_EPS="xcts_utf_gt_item_eps";
    TASK_XCTS_GT_ITEM_FEAT_EPS = "xcts_feat_gt_item_eps";

    TASK_XCTS_weight="weights";

class neko_task_processing(neko_task_processing_meta):
    TYPE="type";
    TYPE_PARAM="type_param";
    TRAINING_EP_PRFX="training_ep_prfx";
    INFR_EP_PRFX = "inference_ep_prfx";
    MOD_PRFX = "mod_prfx";
    NAME="name"

    DATA_DATA_PRFX="data_data_prefix";
    META_DATA_PRFX="meta_data_prefix";
    @classmethod
    def get_abstract_testing_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name,task_type):
        return {
            cls.NAME: task_ep_name,
            cls.TYPE: task_type,
            cls.INFR_EP_PRFX: P("inference", task_ep_name),
            cls.TYPE_PARAM: {},
            cls.MOD_PRFX: task_mod_name,
            cls.DATA_DATA_PRFX: data_ep_name,
            cls.META_DATA_PRFX: meta_ep_name
        };
    @classmethod
    def get_abstract_training_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name,task_type):
        tcfg= cls.get_abstract_testing_task(data_ep_name, meta_ep_name, task_ep_name, task_mod_name, task_type);
        tcfg[cls.TRAINING_EP_PRFX]= P("training",task_ep_name);
        return tcfg;

    @classmethod
    def get_default_single_char_oscr_testing_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name):
        return cls.get_abstract_testing_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name, cls.TASK_TYPE_SINGLE_CLS)

    @classmethod
    def get_default_single_item_oscr_training_task(cls, data_ep_name,meta_ep_name,task_ep_name,task_mod_name):
        return cls.get_abstract_training_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name,cls.TASK_TYPE_SINGLE_CLS);

    @classmethod
    def get_default_seq_testing_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name,maxT):
        tcfg= cls.get_abstract_testing_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name, cls.TASK_TYPE_SEQ);
        tcfg[cls.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_MAXT]=maxT;
        return tcfg;

    @classmethod
    def get_default_seq_training_task(cls, data_ep_name,meta_ep_name,task_ep_name,task_mod_name,maxT):
        tcfg= cls.get_abstract_training_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name,cls.TASK_TYPE_SEQ);
        tcfg[cls.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_MAXT] = maxT;
        tcfg[cls.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_ORDERED]=True;
        return tcfg;
    # collect features vec by utf and do cosine sim alignment of these of same class.
    @classmethod
    def get_default_xtcs_training_task(cls, seq_prfxs,item_prfxs, frozen_item_pfrxs, gtdict_path,task_ep_name,weight):
        tcfg= {
            cls.NAME: task_ep_name,
            cls.TYPE: cls.TASK_TYPE_XTCS,
            cls.INFR_EP_PRFX: P("inference", task_ep_name),
            cls.TYPE_PARAM: {
                neko_xcts_task_param.TASK_XCTS_GTDICTS:gtdict_path,
                neko_xcts_task_param.TASK_XCTS_SEQ_FEAT_EPS: [VN.ROI_FEAT_SEQ(seq_prfx) for seq_prfx in seq_prfxs],
                neko_xcts_task_param.TASK_XCTS_SEQ_UTF_EPS: [VN.GT_TOK_UTF(seq_prfx) for seq_prfx in seq_prfxs],
                neko_xcts_task_param.TASK_XCTS_ITEM_FEAT_EPS: [VN.I_FEAT_VEC(item_prfx) for item_prfx in item_prfxs],
                neko_xcts_task_param.TASK_XCTS_ITEM_UTF_EPS: [VN.UTF(item_prfx) for item_prfx in item_prfxs],
                neko_xcts_task_param.TASK_XCTS_GT_ITEM_FEAT_EPS: [VN.I_FEAT_VEC(frozen_item_pfrx) for frozen_item_pfrx in frozen_item_pfrxs],
                neko_xcts_task_param.TASK_XCTS_GT_ITEM_UTF_EPS: [VN.UTF(frozen_item_pfrx) for frozen_item_pfrx in frozen_item_pfrxs],
                neko_xcts_task_param.TASK_XCTS_weight: weight
            },
            cls.MOD_PRFX: task_ep_name,
        };
        return tcfg;


    @classmethod
    def get_default_semseg_training_task(cls, data_ep_name,meta_ep_name,task_ep_name,task_mod_name):
        return cls.get_abstract_training_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name,cls.TASK_TYPE_SEMSEG);

    @classmethod
    def get_default_ca_insseg_training_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name,maxT,ordered):
        tcfg= cls.get_abstract_training_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name, cls.TASK_TYPE_INSSEG_CA);
        tcfg[cls.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_MAXT] = maxT;
        tcfg[cls.TYPE_PARAM][neko_seq_task_param.TASK_SEQ_ORDERED]=ordered;
        return tcfg;
    @classmethod
    def get_default_semseg_testing_task(cls, data_ep_name, meta_ep_name, task_ep_name,task_mod_name):
        return cls.get_abstract_testing_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name, cls.TASK_TYPE_SEMSEG);

    @classmethod
    def get_default_ordered_ca_insseg_testing_task(cls, data_ep_name,meta_ep_name,task_ep_name,task_mod_name):
        return cls.get_abstract_training_task( data_ep_name, meta_ep_name, task_ep_name,task_mod_name,cls.TASK_TYPE_INSSEG_CA);

