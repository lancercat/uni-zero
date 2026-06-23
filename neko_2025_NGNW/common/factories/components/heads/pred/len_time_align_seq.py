from osocrNG.modular_agents_ocrNG.aggrate_agents.len_based_featseq_flatten import neko_word_flatten_agent

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awn
from neko_sdk.neko_framework_NG.names import P
from osocrNG.modular_agents_ocrNG.metric_agents.neko_lcs_ned_agent import neko_ed_agent
from osocrNG.modular_agents_ocrNG.losses.lenpred_loss import get_per_inst_len_loss_agent_mk2
from neko_sdk.neko_framework_NG.agents.utils.ops import neko_list_sum_reduce_agent
from osocrNG.modular_agents_ocrNG.losses.cls_loss import get_per_inst_ocr_cls_loss_agent_mk2

from neko_sdk.seq2seq.gt_trim import neko_list_trim_maxT_alot_agent
from osocrNG.modular_agents_ocrNG.ocr_label_agents_g2.neko_ccd_label_making_agent import \
     neko_ccd_label_making_agent_mk2s
from osocrNG.trainable_lossNG.os_clsloss import osclsNG_perinstance,osclsNG_perinstance_d,oslenlossNG_perinst

 # get_translate_agent
from osocrNG.modular_agents_ocrNG.pred_subs.lpos_token_translator import lpos_translate_agent_mk2
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.agents.ned_loss_mix_reward import  neko_loss_agent_32x
from neko_sdk.OSR.classification.agents.neko_missing_token_to_unk import neko_missing_to_unk
from osocrNG.modular_agents_ocrNG.att_agents.lpred_agent import neko_basic_lpred_attention_onemod
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.ohem_log_weighting_mk3 import logweighting_loss_agent_mk3_detach_weight_ohem_01_eff


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from osocrNG.ocr_modules_NG.lenpred.lenpred_basic_mk1 import lenpred_basic_mk1


class position_wise_with_length_pred_mod_factory(virtual_mod_factory):

    len_pred_engine=lenpred_basic_mk1;
    PARAM_maxT="maxT";
    def config_local_len_pred(this,modcfgdict,name,maxT):
        e = this.len_pred_engine;
        modcfgdict = this.config_saveable(
            modcfgdict, e, {
                e.PARAM_input_channels: this.gmodcfg.tfe_chs,
                e.PARAM_maxT: maxT+1,
            }, name
        );
        return modcfgdict;
    def config_loss(this,modcfgdict,prfx):
        modcfgdict = this.config_non_saveable(modcfgdict,osclsNG_perinstance,{},
            MN.PER_INSTANCE_OCR_CLS_LOSS_NAME(prfx));
        modcfgdict = this.config_non_saveable(modcfgdict, oslenlossNG_perinst, {},
                                              MN.PER_INSTANCE_OCR_LPRED_LOSS(prfx));
        return modcfgdict;
    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        localmodprfx=mod_prfx_dict[this.MOD_PRFX_local];
        maxT=neko_get_arg(this.PARAM_maxT,params);
        modcfg=this.config_local_len_pred(modcfg,MN.WORD_LPRED(localmodprfx),maxT);
        modcfg=this.config_loss(modcfg,localmodprfx); # always config the loss module--- bcs they don't bite.
        # meta classifier ist not upto the heads' duty to construct and manage--- the meta manager do that
        return modcfg,bogocfg
class  abstract_position_wise_with_length_pred_agt_factory(virtual_agt_factory):
    DATA_PRFX_roiseq_feat = "temproal_feature";
    DATA_PRFX_pred_endpoint = "flatten_temproal_feature_inference"; # training module needs inference prediction for reward computation.
    DATA_PRFX_meta = "meta";

    def arm_flatten_core(this, lacfg,feat_data_prefix,seq_prefix,decode_length_tensor_name):
        lacfg=awa.append_agent_to_cfg(lacfg,"flatten",
                            neko_word_flatten_agent.get_agtcfg(
                                          VN.DENSE_CLS_PRED_LOGIT(feat_data_prefix), decode_length_tensor_name,
                                          VN.FLATTEN_ROI_LOGIT_SEQ(seq_prefix),
                                          VN.FLATTEN_MAP(seq_prefix))
        );
        return lacfg;
    def arm_translate_core(this,lacfg,logit_prefix,meta_data_prefix,decode_length_tensor_name):
        lacfg=awa.append_agent_to_cfg(lacfg, "translation", lpos_translate_agent_mk2.get_agtcfg(
            decode_length_tensor_name,VN.FLATTEN_ROI_LOGIT_SEQ(logit_prefix),VN.TDICT(meta_data_prefix),
            VN.PRED_TEXT(logit_prefix),VN.PRED_TOK(logit_prefix)));
        return lacfg;

    def arm_pred_translate_core(this, lacfg, feat_data_prefix,seq_prefix,logit_prefix,meta_data_prefix,meta_mod_prefix,decode_length_tensor_name):
        lacfg=this.arm_flatten_core(lacfg,feat_data_prefix, seq_prefix, decode_length_tensor_name);
        lacfg =this.arm_translate_core(lacfg,seq_prefix,meta_data_prefix, decode_length_tensor_name);
        return lacfg;

class position_wise_with_length_pred_agt_factory(abstract_position_wise_with_length_pred_agt_factory):


    # DATA_PRFX_flatten_feat_training = "flatten_temproal_feature_training";
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        meta_data_prfx = data_prfx_dict[this.DATA_PRFX_meta];

        feat_seq_data_prfx = data_prfx_dict[this.DATA_PRFX_roiseq_feat];
        head_data_prfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);

        lacfg = awa.empty([VN.SAM_ID(feat_seq_data_prfx),VN.PROTO(meta_data_prfx)]);

        lacfg = awa.append_agent_to_cfg(lacfg, "lpred", neko_basic_lpred_attention_onemod.get_agtcfg(
            VN.WORD_TEMP_FEAT_GLOBAL(feat_seq_data_prfx),
            VN.WORD_LEN_PRED_LOGIT(head_data_prfx), VN.WORD_LEN_PRED_VALUE(head_data_prfx),
            MN.WORD_LPRED(modprfx)
        ));
        # prediction using predicted length has to be done-- bcs you need ned for reward.
        # factory is used to group functionality, is not designed to be modularized unless there is a possibilty for choice.
        # this class is meant to model an entire choice.
        lacfg = this.arm_pred_translate_core(lacfg, head_data_prfx, head_data_prfx, head_data_prfx, meta_data_prfx,
                              modprfx,VN.WORD_LEN_PRED_VALUE(head_data_prfx));

        return awa.append_agent_to_cfg(agtcfg, AN.PRED_LO(agt_prfx), lacfg);

class position_wise_with_length_pred_training_agt_factory(abstract_position_wise_with_length_pred_agt_factory):
    DATA_PRFX_gt = "gt_data_prfx";
    DATA_PRFX_flatten_logit_training = "flatten_logit_training";
    def get_mkgt(this, head_training_prfx,infer_prfx, feat_prfx, gtprfx,meta_prfx,maxT=32):
        lacfg=awn.empty();
        lacfg=awn.append_agent_to_cfg(lacfg,"get_gt_utf",
            neko_list_trim_maxT_alot_agent.get_agtcfg(
                [VN.GT_TOK_UTF(feat_prfx)],
                [VN.GT_TOK_UTF(head_training_prfx)],maxT)
        );
        lacfg=awn.append_agent_to_cfg(lacfg,"decorate_unk",neko_missing_to_unk.get_agtcfg(
            VN.GT_TOK_UTF(head_training_prfx),VN.TDICT(meta_prfx),
            VN.GT_TOK_UTF_WUNK(head_training_prfx)));
        lacfg=awn.append_agent_to_cfg(lacfg,"get_tensor_gt",
            neko_ccd_label_making_agent_mk2s.get_agtcfg(
                VN.PROTO(meta_prfx),VN.GT_TOK_UTF_WUNK(head_training_prfx),VN.TDICT(meta_prfx),
                VN.FLATTEN_ALIGNED_TLABEL(head_training_prfx),VN.WORD_LEN_GT_VALUE(head_training_prfx)
        ));
        # it does not use plabel--- reduction is done in head.
        return lacfg

    def get_head_perinst_loss(this, head_training_prfx,head_infer_prfx, loss_mod_prfx):
        return {
            "agent": awa,
            "params": {
                "agent_list": ["lenpredloss", "clsloss", "add"],
                "lenpredloss": get_per_inst_len_loss_agent_mk2(
                    VN.WORD_LEN_GT_VALUE(head_training_prfx),
                    VN.WORD_LEN_PRED_LOGIT(head_infer_prfx),
                    VN.LEN_LOSS_PER_INST(head_training_prfx),
                    MN.PER_INSTANCE_OCR_LPRED_LOSS(loss_mod_prfx)
                ),
                "clsloss" : get_per_inst_ocr_cls_loss_agent_mk2(
                                                                      VN.FLATTEN_ALIGNED_TLABEL(head_training_prfx),
                                                                      VN.FLATTEN_ROI_LOGIT_SEQ(head_training_prfx),
                                                                      VN.FLATTEN_MAP(head_training_prfx),
                                                                      VN.CLS_LOSS_PER_INST(head_training_prfx),
                                                                      MN.PER_INSTANCE_OCR_CLS_LOSS_NAME(loss_mod_prfx)),
                "add": neko_list_sum_reduce_agent.get_agtcfg(
                    [VN.LEN_LOSS_PER_INST(head_training_prfx), VN.CLS_LOSS_PER_INST(head_training_prfx)],
                    VN.LOSS_PER_INST(head_training_prfx))
                }
        }
    def get_head_perinst_penalty(this,head_training_prfx,head_inference_prfx):
        # loss is just loss. If you see better perf then go for that.
        return {
            "agent": awn,
            "params": {
                "agent_list": ["ned", "penalty"],
                "ned": neko_ed_agent.get_agtcfg(
                                                         VN.GT_TOK_UTF_WUNK(head_training_prfx),VN.PRED_TOK(head_inference_prfx),
                                                         VN.PRED_NED(head_training_prfx)),
                "penalty": neko_loss_agent_32x.get_agtcfg(
                    VN.LOSS_PER_INST(head_training_prfx),
                    VN.PRED_NED(head_training_prfx),
                    VN.PENALTY_PER_INST(P(head_training_prfx,"raw"))),
            }
        };

    def get_reward_and_loss(this,feat_data_prefix,head_training_prfx,head_inference_prfx,meta_prfx,global_gt_prfx,loss_mod_prfx):
        # no sample no loss-- but to make sure all gpus effort goes to smth, gt must present if there is a sample.
        lacfg=awa.empty([VN.SAM_ID(feat_data_prefix)]);
        lacfg=awa.append_agent_to_cfg(lacfg,"mkgt",
            this.get_mkgt(head_training_prfx,head_inference_prfx,feat_data_prefix,global_gt_prfx,meta_prfx));
        lacfg=this.arm_flatten_core(lacfg,head_inference_prfx,head_training_prfx, VN.WORD_LEN_GT_VALUE(head_training_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"mkloss",this.get_head_perinst_loss(head_training_prfx,head_inference_prfx,loss_mod_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"mkreward",this.get_head_perinst_penalty(head_training_prfx, head_inference_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"loss_weighting",
              logweighting_loss_agent_mk3_detach_weight_ohem_01_eff.get_agtcfg(
                  VN.LOG_LIKLYHD(head_inference_prfx),VN.LOSS_PER_INST(head_training_prfx),VN.TASK_LOSS(head_training_prfx),0));
        return lacfg;

    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        meta_prfx = data_prfx_dict[this.DATA_PRFX_meta];
        te_flatten_seq_prfx = neko_get_arg(this.DATA_PRFX_pred_endpoint, data_prfx_dict);

        feat_seq_data_prfx = data_prfx_dict[this.DATA_PRFX_roiseq_feat];
        head_data_training_prfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        gt_prfx=data_prfx_dict[this.DATA_PRFX_gt];
        return awa.append_agent_to_cfg(agtcfg, AN.PRED_LO_TR(agt_prfx), this.get_reward_and_loss(
            feat_seq_data_prfx,head_data_training_prfx,te_flatten_seq_prfx,meta_prfx,gt_prfx,modprfx
        ));

