from osocrNG.modular_agents_ocrNG.aggrate_agents.feat_aggr_agents import neko_word_aggr,neko_word_aggr_mk2,neko_word_aggr_mk2l,neko_word_flatten_agent

def get_temporal_aggr_mk2l(word_feature_name,length_name,attention_map_name,full_feat_seq_name,seq_mod_name,capmaxT=9999):
    e=neko_word_aggr_mk2l;
    cfg={
        "agent":e,
        "params":{
            "iocvt_dict": {
                e.INPUT_length_name:length_name,
                e.INPUT_feature_name:word_feature_name,
                e.INPUT_attention_map_name:attention_map_name,
                e.OUTPUT_feat_seq_name:full_feat_seq_name,
            },
            "modcvt_dict": {
                e.MOD_seq: seq_mod_name,
            },
            e.PARAM_maxT: capmaxT
        }
    }
    return cfg;

def get_seq_flatten(full_feat_seq_name,length_name,flatten_seq_name,mapping_name):
    e=neko_word_flatten_agent;
    cfg = {
        "agent": e,
        "params": {
            "iocvt_dict": {
                e.INPUT_feat_seq_name: full_feat_seq_name,
                e.INPUT_length_name: length_name,
                e.OUTPUT_flatten_mapping_name: mapping_name,
                e.OUTPUT_flatten_feat_seq_name: flatten_seq_name,
            },
            "modcvt_dict": {
            }
        }
    }
    return cfg;