
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awn
from neko_sdk.neko_framework_NG.names import P
from osocrNG.modular_agents_ocrNG.metric_agents.neko_lcs_ned_agent import neko_ed_agent

from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from osocrNG.modular_agents_ocrNG.ocr_label_agents_g2.neko_ccd_label_making_agent import \
     neko_ccd_label_making_agent_mk2s
from osocrNG.trainable_lossNG.os_clsloss import osclsNG_perinstance,oslenlossNG_perinst

 # get_translate_agent
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.agents.ned_loss_mix_reward import  neko_loss_agent_32x
from neko_sdk.OSR.classification.agents.neko_missing_token_to_unk import neko_missing_to_unk
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.ohem_log_weighting_mk3 import logweighting_loss_agent_mk3_detach_weight_ohem_01_eff
from osocrNG.modular_agents_ocrNG.losses.ctc_loss import ctc_loss_agent_mk2

from osocrNG.modular_agents_ocrNG.pred_subs.ctc_token_translator import ctc_translate_agent_mk2, \
    ctc_translate_agent_mk2_utf

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa



class ctc_mod_factory(virtual_mod_factory):
    def config_loss(this,modcfgdict,prfx):

        modcfgdict = this.config_non_saveable(modcfgdict,osclsNG_perinstance,{},
            MN.PER_INSTANCE_OCR_CLS_LOSS_NAME(prfx));
        modcfgdict = this.config_non_saveable(modcfgdict, oslenlossNG_perinst, {},
                                              MN.PER_INSTANCE_OCR_LPRED_LOSS(prfx));
        return modcfgdict;
    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        localmodprfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_loss(modcfg,localmodprfx); # always config the loss module--- bcs they don't bite.
        return modcfg,bogocfg
class ctc_pred_agt_factory(virtual_agt_factory):
    DATA_PRFX_roiseq_feat = "roiseq_feature";
    DATA_PRFX_pred_endpoint = "flatten_roiseq_feature_inference"; # training module needs inference prediction for reward computation.
    DATA_PRFX_meta = "meta";
    def arm_pred_translate_core(this, lacfg,seq_prefix,logit_prefix,meta_data_prefix,meta_mod_prefix):

        lacfg=awa.append_agent_to_cfg(
            lacfg, "translation", ctc_translate_agent_mk2.get_agtcfg(
                VN.DENSE_CLS_PRED_LOGIT(logit_prefix),VN.TDICT(meta_data_prefix),
                VN.PRED_TOK_DENSE(logit_prefix),VN.PRED_TOK(logit_prefix),VN.PRED_TEXT(logit_prefix),0));
        lacfg = awa.append_agent_to_cfg(
            lacfg, "translation_cased", ctc_translate_agent_mk2_utf.get_agtcfg(
                VN.DENSE_CENTER_PRED_LOGIT(logit_prefix), VN.UTF(meta_data_prefix),
                VN.PRED_TOK_DENSE(P(logit_prefix,"cased")), VN.PRED_TOK(P(logit_prefix,"cased")), VN.PRED_TEXT(P(logit_prefix,"cased")), 0));

        return lacfg;


    # DATA_PRFX_flatten_feat_training = "flatten_roiseq_feature_training";
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        meta_data_prfx = data_prfx_dict[this.DATA_PRFX_meta];

        feat_seq_data_prfx = data_prfx_dict[this.DATA_PRFX_roiseq_feat];
        head_data_prfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);

        lacfg = awa.empty([VN.SAM_ID(feat_seq_data_prfx),VN.PROTO(meta_data_prfx)]);

        # prediction using predicted length has to be done-- bcs you need ned for reward.
        # factory is used to group functionality, is not designed to be modularized unless there is a possibilty for choice.
        # this class is meant to model an entire choice.
        lacfg = this.arm_pred_translate_core(lacfg,feat_seq_data_prfx, head_data_prfx, meta_data_prfx,modprfx);
        # copy its id and probability and meta_utf to endpoint. This version does not choose meta, so there is no routing involved.
        # only copy id if routing. let's move this functionality out of this factory.
        return awa.append_agent_to_cfg(agtcfg, AN.PRED_LO(agt_prfx), lacfg);

class ctc_training_extra_agt_factory(ctc_pred_agt_factory):
    DATA_PRFX_gt = "gt_data_prfx";
    def get_mkgt(this, head_training_prfx, feat_prfx,meta_prfx, gtprfx):
        lacfg=awn.empty();
        lacfg=awn.append_agent_to_cfg(lacfg,"get_gt_utf",
            neko_symbol_link_alot_agent.get_agtcfg(
                [VN.GT_TOK_UTF(feat_prfx)],
                [VN.GT_TOK_UTF(head_training_prfx)])
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
                "agent_list": [ "ctcloss"],
                "ctcloss": ctc_loss_agent_mk2.get_agtcfg(
                    VN.FLATTEN_ALIGNED_TLABEL(head_training_prfx),
                    VN.DENSE_CLS_PRED_LOGIT(head_training_prfx),
                    VN.WORD_LEN_GT_VALUE(head_training_prfx),
                    VN.LOSS_PER_INST(head_training_prfx)
                    )
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
        lacfg=awa.empty();
        lacfg=awa.append_agent_to_cfg(lacfg,"mkgt",
            this.get_mkgt(head_training_prfx,feat_data_prefix,meta_prfx,global_gt_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"mkloss",this.get_head_perinst_loss(head_training_prfx,head_inference_prfx,loss_mod_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"mkreward",this.get_head_perinst_penalty(head_training_prfx, head_inference_prfx));
        lacfg=awa.append_agent_to_cfg(lacfg,"loss_weighting",
              logweighting_loss_agent_mk3_detach_weight_ohem_01_eff.get_agtcfg(
                  VN.LOG_LIKLYHD(head_inference_prfx),VN.LOSS_PER_INST(head_training_prfx),VN.TASK_LOSS(head_training_prfx),0));
        return lacfg;

    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        meta_data_prfx = data_prfx_dict[this.DATA_PRFX_meta];
        te_flatten_seq_prfx = neko_get_arg(this.DATA_PRFX_pred_endpoint, data_prfx_dict);
        feat_seq_data_prfx = data_prfx_dict[this.DATA_PRFX_roiseq_feat];
        head_data_training_prfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        gt_prfx=data_prfx_dict[this.DATA_PRFX_gt];
        return awa.append_agent_to_cfg(agtcfg, AN.PRED_LO_TR(agt_prfx), this.get_reward_and_loss(
            feat_seq_data_prfx,head_data_training_prfx,te_flatten_seq_prfx,meta_data_prfx,gt_prfx,modprfx
        ));

# eventually we make routing a mixin
class ctc_pred_agt_factory_routing(ctc_pred_agt_factory):
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        modprfx = mod_prfx_dict[this.MOD_PRFX_local];
        meta_data_prfx = data_prfx_dict[this.DATA_PRFX_meta];

        feat_seq_data_prfx = data_prfx_dict[this.DATA_PRFX_roiseq_feat];
        head_data_prfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);

        lacfg = awa.empty([VN.SAM_ID(feat_seq_data_prfx),VN.PROTO(meta_data_prfx)]);

        # prediction using predicted length has to be done-- bcs you need ned for reward.
        # factory is used to group functionality, is not designed to be modularized unless there is a possibilty for choice.
        # this class is meant to model an entire choice.
        lacfg = this.arm_pred_translate_core(lacfg,feat_seq_data_prfx, head_data_prfx, meta_data_prfx,modprfx);
        # copy its id and probability and meta_utf to endpoint. This version does not choose meta, so there is no routing involved.
        # only copy id if routing. let's move this functionality out of this factory.
        lacfg = awa.append_agent_to_cfg(lacfg, "copy_id_prob", neko_symbol_link_alot_agent.get_agtcfg(
            [VN.LOG_LIKLYHD(feat_seq_data_prfx), VN.SAM_ID(feat_seq_data_prfx),VN.TDICT(meta_data_prfx),VN.DSCRGED(feat_seq_data_prfx)],
            [VN.LOG_LIKLYHD(head_data_prfx), VN.SAM_ID(head_data_prfx),VN.TDICT(head_data_prfx),VN.DSCRGED(head_data_prfx)]
        ));
        return awa.append_agent_to_cfg(agtcfg, AN.PRED_LO(agt_prfx), lacfg);

