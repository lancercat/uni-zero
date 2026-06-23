# you are still free to use strings,
# this is just to guardrail the strings don't get all over the place.
from neko_sdk.neko_framework_NG.names import default_variable_names,default_param_names

def prefix(cls, prefix, term):
    return prefix + term;


class default_ocr_param_names(default_param_names):
    anchors="anchors";
    anchor_names="names";
    anchor_ratio="ratio";

class default_ocr_variable_names(default_variable_names):
    word_feature="word_feature";
    word_feature_name="word_feature_name";

    attention_map="attention_map";
    attention_map_name="attention_map_name";

    selector_name="selector_name";
    selector="selector";
    feat_seq_name="feat_seq_name";
    feat_seq="feat_seq";

    logit_name="logit_name";
    logit="logit";

    length_name="length_name";

    len_pred_logits_name="len_pred_logits_name";
    len_pred_logits="len_pred_logits";



    gt_length_name="gt_length_name";
    gt_length="gt_len";

    tensor_gt_length_name="tensor_gt_length_name";

    tensor_label="tensor_label";
    tensor_label_name="tensor_label_name"

    tensor_global_label="tensor_global_label";
    tensor_global_label_name="tensor_global_label_name"



    pred_length_name="pred_length_name";

    pred_length="len_pred";

    pred_text_name="pred_text_name";
    pred_text="pred_text";

    len_pred_argmax_name="len_pred_argmax_name";
    len_pred_argmax="len_pred_argmax";


class default_ocr_training_variable_names:
    pass;


class default_ocr_testing_variable_names:
    selector_name="selector_name";

