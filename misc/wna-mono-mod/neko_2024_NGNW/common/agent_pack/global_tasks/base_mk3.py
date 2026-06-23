from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent as awa
from osocrNG.modular_agents_ocrNG.att_agents.lpred_agent import get_neko_basic_lpred_attention_onemod
from neko_2024_NGNW.nets_v5.common.agent_pack.loss_agts import get_lenpred_loss
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.CnC.rewards.supervise_on_min_loss import get_neko_supervised_on_min_penalty
from neko_sdk.CnC.rewards.reward_on_exp_loss import get_nekolog_exp_penalty

class neko_base_global_tasks_mk3:

    MN=mod_names;
    VAN=agent_var_names;

    CMD_delay_e = 0;
    CMD_delay_b = 0;  # Learn router from the first iter
    def get_routing_subjects(this,prefix):
        return [prefix + this.VAN.RAW_IMG_NAME,
         prefix + this.VAN.RAW_IMG_TAG,
         prefix + this.VAN.ROUTER_FEAT_NAME,
         prefix + this.VAN.ROUTER_FEATMAP_NAME
         ];
    def append_global_task(this,gtc, prefix, anchor_dict):

        broadcasting_subject = [prefix+this.VAN.TDICT,prefix+this.VAN.PROTO_LABEL,prefix + this.VAN.PROTO_VEC];
        routing_subject = this.get_routing_subjects(prefix);

        return gtc, routing_subject, broadcasting_subject



    def arm_reward(this, ac, anchor_dict, prefix):
        sac = get_neko_supervised_on_min_penalty(
            this.VAN.ROUTER_SAM_ID, this.VAN.PENALTY_PER_INSTANCE,
            this.VAN.ROUTER_LOGIT, this.VAN.ACTUAL_BEST_NAME,
            this.VAN.COMMANDER_LOSS, anchor_dict["names"],
            this.CMD_delay_e, this.CMD_delay_b); # well stalling this can cause some semi-random routing behaviour in the beginning
        ac = awa.append_agent_to_cfg(ac, this.VAN.ANAME_reward_loss, sac);
        return ac;

    def arm_global_loss(this, ac, anchor_dict, prefix):

        ac = this.arm_reward(ac, anchor_dict, prefix);
        ac = this.arm_global_task_losses(ac, anchor_dict, prefix);
        return ac;
        ### training
    def arm_global_task_losses(this, ac, anchor_dict, prefix):
        return ac;
class neko_base_global_tasks_mk3_glp(neko_base_global_tasks_mk3):
    def get_global_task(this,gtc, prefix, anchor_dict):
        gtc, routing_subject, broadcasting_subject=super().append_global_task(gtc,prefix, anchor_dict);
        gtc=awa.append_agent_to_cfg(gtc, "length_prediction",
                                                      get_neko_basic_lpred_attention_onemod(
                                                          prefix + this.VAN.ROUTER_FEAT_NAME,
                                                          prefix + this.VAN.GLOBAL_ATT_LEN_PRED,
                                                          prefix + this.VAN.GLOBAL_ATT_LEN_PRED_AMAX,
                                                          prefix + this.MN.WORD_LEN_PRED));

        return gtc, routing_subject, broadcasting_subject
    def arm_global_task_losses(this, ac, anchor_dict, prefix):
        ac = awa.append_agent_to_cfg(ac,"global_len_loss",get_lenpred_loss(
            prefix+this.VAN.GLOBAL_ATT_LEN_PRED,prefix+this.VAN.TEN_GT_LEN,prefix+this.VAN.LEN_LOSS_PER_INSTANCE,
            this.VAN.LEN_LOSS,this.MN.PER_INSTANCE_OCR_LPRED_LOSS_NAME));
        return ac;
from neko_sdk.CnC.rewards.penalty_collector import neko_penalty_collection_agent
# the policy gradient approach
class neko_base_global_tasks_mk3_log_exp(neko_base_global_tasks_mk3):

    def arm_reward(this, ac, anchor_dict, prefix):
        # ac=awa.append_agent_to_cfg(
        #     neko_penalty_collection_agent.get_agtcfg(
        #         prefix+this.VAN.ROUTER_SAM_ID,prefix+this.VAN.PENALTY_PER_INSTANCE,prefix+this.VAN.ROUTER_LOGIT,
        #         prefix+this.VAN.PENALTY_SPACE,prefix+this.VAN.PENALTY_SPACE_mask,anchor_dict["names"],9999),
        #     "gather_penalty"
        # )
        for a in anchor_dict["names"]:
            ac=awa.append_agent_to_cfg(ac,"penalty_"+a,get_nekolog_exp_penalty(prefix+this.VAN.RAW_IMG_NAME,prefix+a+this.VAN.ROUTER_PATH_LOG_PROB_NAME,prefix+a+this.VAN.PENALTY_PER_INSTANCE,
                prefix+a+this.VAN.COMMANDER_LOSS)); # well stalling this can cause some semi-random routing behaviour in the beginning
        return ac;