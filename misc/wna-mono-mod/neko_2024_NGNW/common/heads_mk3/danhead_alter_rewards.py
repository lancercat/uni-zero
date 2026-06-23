from neko_2024_NGNW.common.heads_mk3.danhead import dan_head_factory_mk3
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent_nograd as awa_ng

from neko_2024_NGNW.common.penalties import neko_perinst_ned_agent,neko_ned_thresholding_mix_agent
from osocrNG.configs.typical_agent_setups.aggr import get_temporal_aggr_mk2l,get_seq_flatten
from osocrNG.configs.typical_agent_setups.os_pred import get_pred_agent, get_translate_agent

class dan_head_factory_mk3_nedmix(dan_head_factory_mk3):
    def get_head_perinst_penalty(this,prefix,head_name):
        # loss is just loss. If you see better perf then go for that.
        return {
        "agent": awa_ng,
        "params": {
            "agent_list": [ "flatten","prediction","translate", "ned","penalty"],
            "flatten": get_seq_flatten(
                prefix + head_name + this.VAN.FULL_WORD_FEAT_SEQ, prefix+head_name+this.VAN.ATT_LEN_PRED_AMAX,
                prefix + head_name + this.VAN.FLATTEN_WORD_FEAT_SEQ_TRNED, prefix + head_name + this.VAN.FLATTEN_MAP_TRNED),
            "prediction": get_pred_agent(prefix + head_name + this.VAN.FLATTEN_WORD_FEAT_SEQ_TRNED,
                                         prefix + this.VAN.PROTO_VEC, prefix + this.VAN.PROTO_LABEL,
                                         prefix + head_name + this.VAN.LOGITS_TRNED,  this.MN.WORD_CLASSIFIER),
            "translate": get_translate_agent(prefix +head_name+ this.VAN.LOGITS_TRNED, prefix+ this.VAN.TDICT,
                                             prefix + head_name + this.VAN.ATT_LEN_PRED_AMAX, prefix +head_name+ this.VAN.PRED_TEXT),
            "ned": neko_perinst_ned_agent.get_agtcfg(prefix +head_name+ this.VAN.LOGITS_TRNED,
                                                     prefix+this.VAN.RAW_WUNK_LABEL_NAME,
                                                     prefix+head_name+this.VAN.PRED_TEXT,
                                                     prefix+head_name+this.VAN.PRED_NED),
            "penalty": neko_ned_thresholding_mix_agent.get_agtcfg(
                prefix+head_name+this.VAN.LOSS_PER_INSTANCE,
                prefix+head_name+this.VAN.PRED_NED,
                prefix+head_name+this.VAN.PENALTY_PER_INSTANCE,
                100,0.6) # NED does not mean damn if it is almost wrong.
        }};
