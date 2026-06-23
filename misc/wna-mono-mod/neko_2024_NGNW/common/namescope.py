class mod_names:
    # Asset names
    # Router
    ## Router common
    ROUTER_FE_name="router_fe";
    ROUTER_AGGR_name = "router_aggr";

    ROUTER_CMD_name="router_command";
    ROUTER_ATT_name="router_attention"
    ROUTER_FE_bbn_name = "router_fe_backbone";
    ## Router train only
    ROUTER_rec_loss_reward = "router_rec_loss_reward";
    ROUTER_WORD_CHAR_LOC_GAU_PRED="router_word_character_location_pred";

    ## Routing control name
    ROUTER_CTRL="router_control";

    # Module pool
    ## GLOBAL Common
    MVN_name="mvn";
    IMG_MVN_name="list_mvn"; # delay the resize to sampling.
    ROUTER_FE_core_name="router_fe_core";

    ## Dropouts --- they may or may not be picked up by any agents, but just having them here won't bite.
    DROP05="drop05";
    DROP03 = "drop03";
    DROP02 = "drop02";
    DROP01="drop01";

    ## Mostly GLOBAL Common, you may want to have a tree or forest,
    # I won't stop you

    FE_FE_core_name="fe_fe_core";

    DTD_name="DTD";
    META_REG="meta_register";

    META_SAM="meta_sampler";

    PROTO_FE = "prototyper_fe";
    PROTO_ATT_FE = "prototyper_att_fe";
    PROTO_AGGR= "prototyper_aggr";

    PROTO_ENC = "prototyper";


    ## Mostly expert specific, so basically used after a prefix.
    COLLATE_name="collate";
    WORD_FE="fe";

    CHAR_ATT="proto_att";
    WORD_ATT="att_mod";
    WORD_GEO_ATT="geo_att_mod";
    EMPTY_CHAR_EMB="empty_char_embedding"; # let's move sptokens out.
    SP_EMB_MOD="sp_token_embedding"; # for gen2 embeddings

    # you don't define sp token locally, bcs it will make management really hard.

    SP_TOK_SEP="[s]";
    SP_TOK_EMP="[-]";


    RECON4x="feat_reconstructor_4x"; # you can not reconstruct word feature with 32x, will yield HUGE image.
    RECON32x="feat_reconstructor_32x"; # for attentioned feat.


    WORD_TEMPORAL_FE="temporal_fe";
    WORD_TEMPORAL_SE="spatial_embedding";

    WORD_CHAR_LOC_GAU_PRED="word_character_location_pred";

    WORD_LEN_PRED="word_len_pred";
    MOE_WORD_LEN_PRED="moe_word_len_pred";
    ATT_SELECTOR="attention_selector";
    WORD_CLASSIFIER="classifier";
    VSB_WORD_CLASSIFIER="vsb_classifier";

    ### training specific modules

    OCR_LOSS_NAME="ocr_loss";

    LANG_MOD_NAME="lang_mod_name"

    PER_INSTANCE_OCR_LOSS_NAME="perinst_ocr_loss";
    PER_INSTANCE_OCR_CLS_LOSS_NAME = "ocr_cls_loss";

    PER_INSTANCE_OCR_LPRED_LOSS_NAME = "ocr_lpred_loss";
    ROUTER_REWARD_NAME="router_reward_name";

    COLLATE_GRID_MKER="collate_grid_maker";

    # ducky related stuff goes here
    VSB_PROTO_CACHE = "global_vis_proto_cache"

    # dict goes with the agent--- eventually tokenization will be delegated with gtdict and an embedding module.
    ABI_LIKE_LM="abi_like_language_model";


    ### LSCT character level goes here;
    LSCT_CHAR_COLLATE_name="lsctc_collate";

    LSCT_CHAR_FE="lsctc_fe";
    LSCT_CHAR_ATT="lsctc_att";
    LSCT_CHAR_AGGR="lsct_char_aggr"; # hey what about we have an agent to do this?


class agent_var_names:
    # endpoint names are controlled by the factories, not set globally here.

    # variable_names
    RAW_BEACON_NAME="raw_beacon";
    RAW_SIZE_NAME="raw_size";
    RAW_ANCHOR_NAME="raw_data_anchor_preference"; # where human thinks the anchors should be, could be None if human is not sure about the routing.
    RAW_IMG_NAME = "raw_image";
    RAW_IMG_NAME_PADDED = "raw_image_padded";
    META_DICT_NAME="meta_dict"; # meta paths involved during training, a dictionary of strings.
                                                  # The users are responsible to setup agents and modules elsewhere
                                                  # manage the cache. Maybe---the user want to use radicals as meta



    RAW_BMASK_NAME="raw_bmask";
    ACTUAL_BEST_NAME="actual_best";
    ROUTER_MASK_NAME_EMPTY="NEP_skipped_NEP";
    ROUTER_MASK_NAME_LEN = "len_masked";
    ROUTER_MASK_NAME_LEN_AR = "len_ar_masked";
    ROUTER_MASK_NAME = "final_routing_masked";

    ROUTER_MASK_NAME_LEN_AR_RULE = "len_ar_masked_rule";

    TENSOR_PROTO_IMG_NAME="tensor_proto_img";
    TENSOR_BEACON_NAME="tensor_beacon";
    ROUTER_FEAT_NAME="router_feature";
    ROUTER_FEATMAP_NAME="router_featmap_name";
    ROUTER_CMD_NAME="router_command";
    ROUTER_ACT_NAME="router_action";
    ROUTER_PATH_NAME="router_paths"; # the specific path
    ROUTER_PATH_ID_NAME="router_path_ids";  # the id of each path. used to gather rewards.
    ROUTER_PATH_LOG_PROB_NAME = "router_path_prob";
    DETACHED_ROUTER_PATH_LOG_PROB_NAME = "detached_router_path_prob";

    ROUTER_LOGIT="router_logit";
    ROUTER_LOGPROB="router_logprob";

    ROUTER_LOSS="router_loss_reward";
    # Even if we want some kind of global UUID,
    # it's necessary to record the in-batch id of the instance
    # (originated from which sample)
    ROUTER_SAM_ID="router_sample_id";
    ROUTER2_SAM_ID="router2_sample_id";

    PROTO_VEC="prototypes";
    RAW_PROTO_VEC="raw_prototypes";


    PROTO_VEC_ROTATED="prototypes_rotated";
    RAW_PROTO_VEC_ROTATED="raw_prototypes_rotated";

    POSSIBLE_ROTATIONS="possible_rotations";


    PROTO_VEC_DETACHED="prototypes_detached";

    PROTO_LABEL="proto_label";
    PROTO_UTF="proto_utf";
    PROTO_GLOBAL_ID="proto_global_id"

    GLOBAL_PROTO_VEC="global_proto_vector";
    GLOBAL_PROTO_UTF="global_proto_utf";

    META_seen_indicator_hack="meta_seen_indicator_hack"; # this will probably go after we have a smarter way to handle multi-meta.

    TDICT="tdict";
    RAW_IMG_TAG= "uid"; # describes the input file name. e.g. "rctw-00001" or something of the same effect.
    GTDICT="gtdict";

    ALIGNED_RAW_IMG_NAME="aligned_raw_image";


    COLLATE_GRID_NAME="collate_grid"; # well if you want some TPS/STN shit, construct this grid yourself.


    TEN_IMG_NAME_UNA="tensor_image_unaligned";
    TEN_IMG_NAME="tensor_image";
    WORD_FEAT="word_feature";
    LAST_WORD_FEAT="last_word_feature";

    WORD_FEAT_DETACHED="word_feature_detached";

    # Either these two will be used depends whether we detach or not.
    WORD_FEAT_DETACHED_SE="word_feature_detached_with_se";
    WORD_FEAT_SE="word_feature_with_se";

    WORD_TEMP_FEAT="word_temporal_feature";
    WORD_TEMP_ENDPOINTS="word_temporal_feature_endpoints";
    WORD_TEMP_ENDPOINTS_GPOOL="word_temporal_global_feature";
    MOE_WORD_TEMP_ENDPOINTS_GPOOL="moe_word_temporal_global_feature";

    FULL_WORD_FEAT_SEQ="full_word_feature_sequence";

    FULL_WORD_FEAT_SEQ_AGO="full_word_feature_sequence_attention_gradient_only";
    FULL_WORD_FEAT_SEQ_FGO="full_word_feature_sequence_feat_gradient_only";
    FULL_WORD_FEAT_DETACHED="full_word_feature_sequence_detached";

    PA_WORD_FEAT_SEQ="paranormal_word_feature_sequence";
    MOE_FULL_WORD_FEAT_SEQ="moe_full_word_feature_sequence";


    FLATTEN_WORD_FEAT_SEQ="flatten_word_feature_sequence";
    FLATTEN_WORD_FEAT_SEQ_DETACHED="flatten_word_feature_sequence_training_detached"; # replaces for ned. Also used for static prediction
    FLATTEN_WORD_FEAT_SEQ_TRNED="flatten_word_feature_sequence_training_for_ned";

    FLATTEN_WORD_FEAT_SEQ_AGO="flatten_word_feature_sequence_attention_gradient_only";
    FLATTEN_WORD_FEAT_SEQ_FGO="flatten_word_feature_sequence_feat_gradient_only";


    FLATTEN_WORD_FEAT_SEQ_UP="flatten_word_feature_sequence_unparted";
    PA_FLATTEN_WORD_FEAT_SEQ="PA_flatten_word_feature_sequence";
    MOE_FLATTEN_WORD_FEAT_SEQ="moe_flatten_word_feature_sequence";

    FLATTEN_MAP="flatten_map";
    FLATTEN_MAP_AGO="flatten_map_attention_map_gradient_only";
    FLATTEN_MAP_FGO="flatten_map_feat_gradient_only";

    FLATTEN_MAP_TRNED="flatten_map_trned";

    MOE_FLATTEN_MAP="moe_flatten_map";

    TEN_GT_LEN="tensor_gt_length";

    ATT_LEN_PRED="attention_len_pred_logits";
    ATT_LEN_PRED_AMAX = "attention_len_pred_logits_argmax";

    MOE_ATT_LEN_PRED="moe_attention_len_pred_logits";
    MOE_ATT_LEN_PRED_AMAX="moe_attention_len_pred_logits_argmax"

    GLOBAL_ATT_LEN_PRED="global_attention_len_pred_logits"
    GLOBAL_ATT_LEN_PRED_AMAX="global_attention_len_pred_logits_argmax"

    ATT_SEL_LOGITS="attention_selection_logits";
    ATT_SEL_PROB="attention_selection_probability";
    ATT_SEL_PROB_DETACHED="attention_selection_probability_detached";

    ATT_SEL_LOSS="attention_selection_loss";

    ATT_MAP="attention_map";
    GEO_ATT_MAP="geo_attention_map";
    ATT_MAP_DETACHED="attention_map_detached";
    ATT_DQUERY="attention_dyna_query";

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2465539/
    ATT_PRED_GAU_PARA_MEAN="predicted_gaussian_mean";
    ATT_PRED_GAU_PARA_VAR = "predicted_gaussian_var";

    PRED_NED="prediction_normalized_editdistance";

    ATT_GAU_PARA_MEAN="attention_fitted_gaussian_mean";
    ATT_GAU_PARA_VAR="attention_fitted_gaussian_variance";
    ATT_GAU_PARA_MEAN_detached="DETACHED_attention_fitted_gaussian_mean";
    ATT_GAU_PARA_VAR_detached="DETACHED_attention_fitted_gaussian_variance";
    ATT_GAU_MSK="reconstructed_attention_mask_from_gaussian_parameter";
    ADHD_LOSS="adhd_loss";
    PA_ALIGNMENT_LOSS="pa_alignment_loss";
    LOGITS="logits";
    # decoded from global_protocol. Since the protovecs may undergo EMA or other weird, non-GD updates, so it's call ducky.


    LOGITS_TRNED="logits_for_training_ned";

    MOE_LOGITS="moe_logits";
    PA_LOGITS="pa_logits";
    PRED_TEXT="pred_text";

    # flatten with gt length, to reduce training variance
    # alternatively, one may choose to hack gt, but that's not what we aim to do here.
    MVB_TRAIN_FLATTEN_WORD_FEAT_SEQ="mvb_train_flatten_word_feature_sequence"
    MVB_TRAIN_LOGITS="mvb_train_logits";
    MVB_TRAIN_FLATTEN_MAP="mvb_train_flatten_map";
    MVB_TRAIN_CLS_LOSS_PER_INSTANCE="mvb_train_cls_loss_per_instance";


    MVB_TEST_FLATTEN_MAP = "mvb_test_flatten_map"
    MVB_TEST_FLATTEN_WORD_FEAT_SEQ="mvb_test_flatten_word_feature_sequence"
    MVB_TEST_DETACHED_FLATTEN_WORD_FEAT_SEQ= "mvb_test_detached_flatten_word_feature_sequence"

    MVB_TEST_DETACHED_FEAT_LOGITS = "mvb_test_detached_logits";
    MVB_TEST_DETACHED_FEAT_PRED_TEXT="mvb_test_detached_pred_dict"
    MVB_TEST_DETACHED_FEAT_PRED_NED="mvb_test_detached_pred_ned"

    # predicted text from some "prototypes" that does not come from GD...


    PA_PRED_TEXT="pa_pred_text";
    # paranormal attention does not (always) exist alone.
    # You don't want to move the ghosts without hints

    COLLECTED_FEAT_SEQ = "collected_expert_feats";
    COLLECTED_LOSS = "collected_expert_losses";

    # we can't avoid LM all the time. It's time to face it in an open world.


    # training variables
    TENSOR_LABEL_NAME="tensor_label";
    TENSOR_GLOBAL_LABEL_NAME="tensor_global_label";
    RAW_LABEL_NAME = "raw_label";
    RAW_WUNK_LABEL_NAME="raw_label_wunk";
    CENSORED_RAW_LABEL_NAME="censored_raw_label"

    LOSS_OCR="ocr_loss";
    DETACHED_LOSS_PER_INSTANCE="detached_loss_perinstance";
    LOSS_PER_INSTANCE="loss_per_instance";
    MOE_CLS_LOSS_PER_INSTANCE="moe_cls_loss_per_instance"
    PA_CLS_LOSS_PER_INSTANCE="pa_cls_loss_per_instance"



    CLS_LOSS_PER_INSTANCE="cls_loss_per_instance";
    DETACHED_CLS_LOSS_PER_INSTANCE="detached_cls_loss_per_instance";

    PENALTY_PER_INSTANCE="penalty_per_instance";
    LEN_LOSS_PER_INSTANCE="len_loss_per_instance";


    PENALTY_SPACE="penalty_space";
    PENALTY_SPACE_mask="penalty_space_mask";


    CLS_LOSS="cls_loss";
    MOE_CLS_LOSS="moe_cls_loss";
    PA_CLS_LOSS="pa_cls_loss";
    LEN_LOSS="len_loss";
    MOE_LEN_LOSS="moe_len_loss";
    LOSS_PER_PATH="loss_per_path";

    EXPERT_ALIGNMENT_LOSS="length_alignment_loss";

    # could be loss, or anything for that matters.
    BRANCH_PENALTY="branch_penalty";

    TOTAL_LOSS="total_loss";
    COMMANDER_LOSS="total_reward";



    VSB_PROTO_VEC="vsb_proto";
    VSB_TDICT="vsb_tdict";
    VSB_PLABEL="vsb_plabel";
    VSB_PRED_TEXT="ducky_pred_text";
    VSB_LOGITS_TRNED="ducky_"+LOGITS+"_for_ned";
    VSB_LOGITS="ducky_"+LOGITS;
    VSB_TENSOR_LABEL_NAME="ducky_tlabel";
    VSB_LOSS="ducky_cls_loss";
    VSB_LOSS_PER_INSTANCE="ducky_loss_per_inst";
    VSB_CLS_LOSS_PER_INSTANCE="ducky_cls_loss_per_inst";

    LM_PROTO_CACHE = "global_semantic_cache";
    LM_CORR_TEXT="lm_corr_text";
    LM_TEXT_LABEL="lm_text_label";
    LM_CORR_LOGITS="lm_corr_logits";  # this is a list, bcs we may have multiple views.
    LM_CORR_LOSS="lm_corr_loss";

    LM_PROTO_VEC_nosp="lm_proto_nosp";
    LM_TDICT_nosp="lm_tdict_nosp";
    LM_PLABEL_nosp="lm_plabel_nosp";

    LM_PROTO_VEC="lm_proto_nosp";
    LM_TDICT="lm_tdict_nosp";
    LM_PLABEL="lm_plabel_nosp";



    SELF_CONTRAST_LOGIT="self_contrast_logit";
    SELF_CONTRAST_LABEL="self_contrast_label";
    SELF_CONTRAST_LOSS_PERINST="self_contrast_loss_perinst";
    SELF_CONTRAST_LOSS_TOTAL="self_contrast_loss";

    RECON_GT="reconstruction_gt";
    RECON_mask="reconstruction_mask";
    RECON_map="reconstructed_map";
    RECON_img="reconstructed_image";
    RECON_loss_perinst="reconstruction_loss_per_inst";
    RECON_loss="reconstruction_loss";

    LSCT_CHAR_tdict="lsctc_tdict";
    LSCT_CHAR_texts = "lsctc_texts";
    LSCT_CHAR_fnts = "lsctc_fnts";
    LSCT_CHAR_plabel = "lsctc_plabel";
    LSCT_CHAR_tensor_labels="lsctc_tensor_labels";
    LSCT_CHAR_raw_img="lsctc_raw_im";
    LSCT_CHAR_TEN_IMG_NAME="lsctc_tensor_im";
    LSCT_CHAR_raw_masks="lsctc_raw_masks";
    LSCT_CHAR_raw_masks_transformed="lsctc_raw_masks_transformed";
    LSCT_CHAR_orient="lsctc_raw_mask_orients";
    LSCT_CHAR_att_msks="lsctc_attention_masks";

    LSCT_CHAR_ALIGNED_RAW_IMG_NAME="lsctc_aligned_raw_img";
    LSCT_TEN_IMG_NAME="lsctc_tensor_img";
    LSCT_CHAR_FEAT_MAPS = "lsctc_featmaps";
    LSCT_CHAR_FEAT_VEC="lsctc_featvecs";
    LSCT_CHAR_logits = "lsctc_logits";
    LSCT_CHAR_loss_name="lsctc_cls_loss";

    STATIC_META_VALID_FLAG="language_valid_flag"

    #queue names
    INCOMING_DQ="data_queue";

    # agent names
    ANAME_mod="mod";
    ANAME_command="command";
    ANAME_mod_loss="mod_loss";

    ANAME_per_instance_loss="perinst_loss";
    ANAME_per_instance_loss_detacher="perinst_loss_detacher";

    ANAME_per_instance_cls_loss="perinst_cls_loss";
    ANAME_PA_per_instance_cls_loss="PA_perinst_cls_loss";

    ANAME_per_instance_cls_loss_detacher="perinst_cls_loss_detacher";
    ANAME_punisher="punisher"; # makes penalty.

    ANAME_per_instance_lenpred_loss="perinst_lenpred_loss";
    ANAME_lenpred_loss="lenpred_loss";

    ANAME_per_instance_lenpred_gt="perinst_lenpred_gt";

    ANAME_cls_loss_aggr="cls_loss_aggr";
    ANAME_PA_cls_loss_aggr="PA_cls_loss_aggr";

    ANAME_lpred_loss_aggr = "lpred_loss_aggr";

    ANAME_reward_collector="reward_collector";
    ANAME_reward_loss="reward_loss";

    ANAME_controller="controller";
    ANAME_training_logger="training_logger";
    ANAME_backward_agent="backward";

    ANAME_core_routine="core_routine";


    ANAME_trainer="trainer";
    ANAME_workspace_init="init_workspace";
    ANAME_aggr="aggregator";
    ANAME_tester="tester";

    ANAME_language_model="language_model";
    ANAME_language_model_gt_maker="language_model";
    ANAME_language_model_loss="language_model_loss";

    DBG_ATT_IMS="debug_attention_images";
    DBG_GT_VPR_PATCHES="debug_gt_vpr_patches";
    DBG_PADDED_RAW_IM="debug_padded_images";
    DBG_MATCHING_FLAIRS= "debug_matching_flairs";
    DBG_ROUTING_FLAIRS= "debug_routing_flairs";
    DBG_RESULT_PANEL="debug_result_panel";