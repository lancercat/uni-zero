from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import get_neko_pool_and_cat_agent
from osocrNG.modular_agents_ocrNG.att_agents.lpred_agent import get_neko_basic_lpred_attention_onemod
from osocrNG.modular_agents_ocrNG.losses.lenpred_loss import get_per_inst_len_loss_agent_mk2
from osocrNG.modular_agents_ocrNG.att_agents.joint_lpred_agent import get_neko_pre_joint_lpred_attention_onemod


class neko_gap_fc_counter:
    MN = mod_names;
    VAN = agent_var_names;
    def get_lenpred(this,prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["tfp","lpred"],
                "tfp": get_neko_pool_and_cat_agent(
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL),
                "lpred": get_neko_basic_lpred_attention_onemod(
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL,
                    prefix + this.VAN.ATT_LEN_PRED,
                    prefix + this.VAN.ATT_LEN_PRED_AMAX,
                    prefix + this.MN.WORD_LEN_PRED),
            }
        }
        return ac;
    def get_len_loss(this,prefix):
        return get_per_inst_len_loss_agent_mk2( prefix + this.VAN.TEN_GT_LEN,  prefix  + this.VAN.ATT_LEN_PRED,
                                                                     prefix + this.VAN.LEN_LOSS_PER_INSTANCE,
                                                                     this.MN.PER_INSTANCE_OCR_LPRED_LOSS_NAME);


class neko_gap_fc_counter_global_local:
    MN = mod_names;
    VAN = agent_var_names;
    def get_lenpred(this,prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["tfp","lpred"],
                "tfp": get_neko_pool_and_cat_agent(
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL),
                "lpred": get_neko_pre_joint_lpred_attention_onemod(
                    [prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL,prefix+this.VAN.ROUTER_FEAT_NAME],
                    prefix + this.VAN.ATT_LEN_PRED,
                    prefix + this.VAN.ATT_LEN_PRED_AMAX,
                    prefix + this.MN.WORD_LEN_PRED),
            }
        }
        return ac;
    def get_len_loss(this,prefix):
        return get_per_inst_len_loss_agent_mk2( prefix + this.VAN.TEN_GT_LEN,  prefix  + this.VAN.ATT_LEN_PRED,
                                                                     prefix + this.VAN.LEN_LOSS_PER_INSTANCE,
                                                                     this.MN.PER_INSTANCE_OCR_LPRED_LOSS_NAME);
