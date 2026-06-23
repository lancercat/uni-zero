from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.names import VR

class VN_PRFX:
    @classmethod
    def SAM(cls,prefix,what):
        return VR(P(prefix,"sample"),what);
    @classmethod
    def GT_TAINT(cls,prefix,what):
        return VR(P(prefix, "gt_taint"), what);
    @classmethod
    def is_eval_safe(cls,what):
        return what.find("gt_taint")==-1;

class VN_THING:

    @classmethod
    def IM(cls):
        return "image";
    @classmethod
    def TENSOR_LIST(cls):
        return "tensor_list";

    @classmethod
    def TENSOR(cls):
        return "tensor";


class VN_STATE:
    @classmethod
    def RAW(cls,what):
        return P(what,"raw");
    @classmethod
    def NORMED(cls,what):
        return P(what,"normed");

    @classmethod
    def CPU(cls,what):
        return P(what,"tensor");

    @classmethod
    def DEV_SIDE(cls,what):
        return P(what,"on_device");


# intended IO
class VNIO:
    @classmethod
    def SAM_IM(this, prefix=""):
        return P(prefix, "sample_images");

    @classmethod
    def SAM_ID(this, prefix=""):
        return P(prefix, "sample_id"); # this is a batchid that you can operate on in a map/reduce operation.
    @classmethod
    def SAM_ORIG(this, prefix=""):
        return P(prefix, "sample_orig"); # this is where you get the path, etc.
    @classmethod
    def SAM_ORIENT(cls,prefix=""):
        return P(prefix, "sample_orientation");
    @classmethod
    def SAM_THUMB(this, prefix=""):
        return P(prefix, "sample_thumbs");

    @classmethod
    def COLL_GRID(this,prefix=""):
        return P(prefix,"collation_grid");

    @classmethod
    def IM_normed_tensor_list(this, prefix=""):
        return P(prefix, "images_normed_tensors_list");
    @classmethod
    def IM_normed_tensor(this, prefix=""):
        return P(prefix, "images_normed_tensors");

    # used in dataloaders, we don't want queued data to hog vram.
    @classmethod
    def IM_normed_tensor_cpu(this, prefix=""):
        return this.IM_normed_tensor(P(prefix,"cpu"));
        return P(prefix, "images_normed_tensors");

    @classmethod
    def SAM_THUMB_normed_tensor_list(this, prefix=""):
        return  this.IM_normed_tensor_list(P(prefix,"thumbnail"));

    @classmethod
    def SAM_THUMB_normed_tensor_cpu(this,prefix=""):
        return this.IM_normed_tensor_cpu(P(prefix,"thumbnail"));

    @classmethod
    def SAM_THUMB_normed_tensor(this, prefix=""):
        return this.IM_normed_tensor(P(prefix, "thumbnail"));


    @classmethod
    def PROTO(cls,prefix=""):
        return P(prefix, "proto_vector");
    # we don't need two plabels
    @classmethod
    def PROTO_LABEL(cls,prefix=""):
        return P(prefix, "proto_label");

    @classmethod
    def SINFO_ID_LST(this, prefix=""):
        return P(prefix, "sinfo_id_list"); # a list of ids, the side info is the id. the id is used to fetch embedding.
    @classmethod
    def SINFO_ID_TEN(this,prefix=""):
        return P(prefix, "sinfo_id_tensor");  # a list of ids, the side info is the id. the id is used to fetch embedding.
    @classmethod
    def IM_raw(this, prefix=""):
        return P(prefix, "images_raw");
    @classmethod
    def IM_tensor_list(this, prefix=""):
        return P(prefix, "image_tensors_list");

    @classmethod
    def UTF(this, prefix=""):
        return P(prefix, "utf");
    @classmethod
    def SAM_FNT(this,prefix=""):
        return P(prefix,"sample_fnt"); # if you want to do style prediction and this field exists

    @classmethod
    def GT_TOK_UTF(this, prefix=""):
        return P(prefix, "sample_utf_tokenized");
        # well in case we have weird tokenizing stuff.
        # tokenizer goes with datasource. at least the first tokenizer goes with data.

    @classmethod
    def GT_TOK_UTF_WUNK(this, prefix=""):
        return P(prefix, "sample_utf_tokenized_withunk");
        # well in case we have weird tokenizing stuff.
        # tokenizer goes with datasource. at least the first tokenizer goes with data.
    @classmethod
    def DBG_GT_VPR_PATCHES(this,prefix=""):
        return  P(prefix, "debug_gt_pred");

    # image tensor converted back to np. used to visualize what is fed to the network
    @classmethod
    def DBG_IMTEN(this, prefix=""):
        return P(prefix, "debug_im_tensor_image");
    @classmethod
    def DBG_SMPRNDR(this, prefix=""):
        return P(prefix, "debug_sample_render");
    @classmethod
    def DBG_CANVAS(this, prefix=""):
        return P(prefix, "debug_canvas");
    @classmethod
    def DBG_MATCHING_FLAIRS(this, prefix):
        return P(prefix, "debug_pred_matching_flair");

    @classmethod
    def SAM_MSK(this, prefix=""):
        return P(prefix, "sample_mask");



    @classmethod
    def PRED_INSSEG_DENSE_MSK_LOGIT(this, prefix=""):
        return  P(prefix, "pred_insseg_dense_cls_msk");
    @classmethod
    def PRED_INSSEG_DENSE_MSK_PROB(this, prefix=""):
        return  P(prefix, "pred_insseg_dense_fg_prob");
    @classmethod
    def GT_INSSEG_MSK(this, prefix=""):
        return VN_PRFX.GT_TAINT(prefix, "sample_instance_mask");
    @classmethod
    def GT_INSSEG_DENSE_MSK_TEN_LST(this, prefix=""):
        return  VN_PRFX.GT_TAINT(prefix, "insseg_dense_cls_msk_tensor_list");
    @classmethod
    def GT_INSSEG_DENSE_MSK_TEN_LST_ALGN(cls, prefix):
        return VN_PRFX.GT_TAINT(prefix, "insseg_dense_cls_msk_tensor_list_aligned");
    @classmethod
    def GT_INSSEG_CBM_IDX(cls, prefix):
        return VN_PRFX.GT_TAINT(prefix, "insseg_cbm_cls_idx");

    @classmethod
    def GT_SEMSEG_DENSE_MSK_TEN_LST(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_cls_msk_tensor_list");
    @classmethod
    def GT_SEMSEG_DENSE_MSK_TEN_LST_ALGN(cls, prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_cls_msk_tensor_list_aligned");
    @classmethod
    def GT_SEMSEG_DENSE_MSK_TEN(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_cls_msk_tensor");
    # when a pixel is 20% apple, 80% an orange, absolute not banana or unknown.
    @classmethod
    def GT_SEMSEG_DENSE_LOGIT_TEN(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_cls_logit_tensor");
    @classmethod
    def GT_SEMSEG_DENSE_CASE_LOGIT_TEN(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_case_logit_tensor");
    @classmethod
    def GT_SEMSEG_DENSE_MSK_VAL_TEN(cls, prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_dense_cls_msk_valid_tensor");

    @classmethod
    def GT_SEMSEG_CBM_MSK(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_cbm_cls_msk");
    @classmethod
    def GT_SEMSEG_CBM_MSK_TEN(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_cbm_cls_msk_tensors");
    @classmethod
    def GT_SEMSEG_CBM_UTF(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_cbm_cls_utf");
    @classmethod
    def GT_INSSEG_UTF(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "insseg_cls_utf");

    @classmethod
    def GT_SEMSEG_CBM_IDX(cls, prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_cbm_cls_idx");
    @classmethod
    def GT_BIN_MSK(this, prefix=""):
        return P(prefix, "sample_binary_mask");
    @classmethod
    def GT_BIN_MSK_TEN(this,prefix=""):
        return P(prefix, "sample_binary_mask_tensor");

    # counts howmany
    @classmethod
    def GT_SEMSEG_CBM_OVR(cls,prefix):
        return VN_PRFX.GT_TAINT(prefix, "semseg_overlap");


    # we believe that there may be a need to keep all
    # for now pls keep the prefix corresponded to the meta part---
    @classmethod
    def SAM_MSK_raw(this, prefix=""):
        return P(prefix, "sample_mask_uncollated");

    @classmethod
    def GT_BIN_MSK_raw(this, prefix=""):
        return VR(this.GT_BIN_MSK(prefix),"uncollated");


    @classmethod
    def SAM_UPFRONT_MSK_BIN(this, prefix=""):
        return VR(this.GT_BIN_MSK(prefix),"upfront");
    @classmethod
    def SAM_UPFRONT_MSK_INS(this, prefix=""):
        return VR(this.GT_INSSEG_MSK(prefix),"upfront");

    @classmethod
    def SAM_UPFRONT_MSK(this, prefix=""):
        return P(prefix, this.SAM_MSK("upfront"));

    @classmethod
    def SAM_UPFRONT_MSK_raw(this, prefix=""):
        return P(prefix, this.SAM_MSK_raw("upfront"));
    @classmethod
    def SAM_ROT(this, prefix=""):
        return P(prefix, "sample_orientation");

    @classmethod
    def FEAT_MAP_LST(cls,prefix=""):
        return P(prefix,"feature_map_list");
    @classmethod
    def FEAT_MAP_LST_SE(cls,prefix=""):
        return P(prefix,"feature_map_list_spatila_embedding");
    @classmethod
    def FEAT_MAP_TAG(cls,prefix=""):
        return P(prefix,"feature_map_tags");

    @classmethod
    def I_FEAT_MAP(cls, prefix=""):
        return P(prefix, "feature_map_instance_endpoint");
    @classmethod
    def I_FEAT_VEC(this, prefix=""):
        return P(prefix, "im_feature_vector");

    @classmethod
    def FEAT_MAP_GAEP(cls, prefix=""):
        return P(prefix, "feature_map_geometric_attention_endpoint");


    @classmethod
    def GLOBAL_ATT_MAP(this, prefix=""):
        return P(prefix, "attentionmap");



    # if meta duplicate, you will want this to make plabel.
    # missing confidence will be.... bah i don't know.
    # maybe we will have 0 or -inf... lemme think



    @classmethod
    def LCS_GT_IDX(cls,prefix):
        return P(prefix,"lcs_gt_idx");


    @classmethod
    def SINFO_UTF_wunk(cls, prefix=""):
        return P(prefix, "sinfo_utf_with_unk");
    @classmethod
    def META_TDICT_PATH(this, prefix=""):
        return P(prefix,
                 "tdict_path");  # endpoint. decorate however you want, make sure that the default one uses this name.
    @classmethod
    def META_TDICT_HASH(this, prefix=""):
        return P(prefix,
                 "tdict_hash");  # endpoint. decorate however you want, make sure that the default one uses this name.



    # data loader default, if merge happens change prefix pls

    @classmethod
    def TDICT_DATA(cls,prefix):
        return P(prefix,"tdict_raw"); # whatever the dataloader knows. handling unknown and sptokens is not dataloaders concern.



    @classmethod
    def UTF_DATA(this, prefix=""):
        return P(prefix, "side_info_utf_raw");
    @classmethod
    def SIDE_INFO_DATA(this, prefix=""):
        return P(prefix, "side_info_collated");

    @classmethod
    def TASKPERF(this, prefix=""):
        return P(prefix,
                 "task_performance");  # endpoint. decorate however you want, make sure that the default one uses this name.

    @classmethod
    def TDICT(this, prefix=""):
        return P(prefix, "tdict"); # endpoint. decorate however you want, make sure that the default one uses this name.
    # as if we use any kind of positionwise decoder we will need aligined flatten logits to compute loss.
    @classmethod
    def DENSE_CENTER_PRED_LOGIT(this, prefix=""):
        return P(prefix, "dense_center_pred_logits");
    # as if we use any kind of positionwise decoder we will need aligined flatten logits to compute loss.
    @classmethod
    def DENSE_CLS_PRED_LOGIT(this, prefix=""):
        return P(prefix, "dense_class_pred_logits");

    @classmethod
    def FLATTEN_PRED_LOGIT(this, prefix=""):
        return P(prefix, "flatten_pred_logits");
    @classmethod
    def PRED_TEXT(this, prefix=""):
        return P(prefix, "pred_text");
    @classmethod
    def PRED_TOK(this, prefix=""):
        return P(prefix, "pred_tokens");
    @classmethod
    def PRED_TOK_DENSE(this, prefix=""):
        return P(prefix, "pred_tokens_dense"); # for debugging---- what does the model read?
    @classmethod
    def FLATTEN_ALIGNED_TLABEL(this, prefix=""):
        return P(prefix, "tensor_label_aligned");

    @classmethod
    def LOSS_PER_INST(this,prefix=""):
        return P(prefix,"loss_per_inst"); # if the modification ain't going troublesome, do it here.

    @classmethod
    def PENALTY_PER_INST(this,prefix=""):
        return  P(prefix,"penalty_per_instance")

    @classmethod
    def PRED_NED(this, prefix=""):
        return P(prefix, "prediction_ned");  # if the modification ain't going troublesome, do it here.
    @classmethod
    def PENALTY(this,prefix=""):
        return  P(prefix,"penalty")


    @classmethod
    def TASK_LOSS(this,prefix=""):
        return P(prefix, "loss_subtotal");  # whatever loss that requires a bp.
