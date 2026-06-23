
from osocrNG.modular_agents_ocrNG.fe_agents.origin_fe import origin_fe_sub


def get_origin_fe(word_tensor_image_name,word_feature_name,word_fe_name):
    return {
        "agent":origin_fe_sub,
        "params":{
            "iocvt_dict": {
                origin_fe_sub.INPUT_image_name:word_tensor_image_name,
                origin_fe_sub.OUTPUT_feature_name:word_feature_name,
            },
            "modcvt_dict": {
                origin_fe_sub.MOD_feature_extractor_name: word_fe_name,
            }
        }
    }

def get_planned_fe(word_tensor_image_name,word_feature_name,word_fe_name):
    return {
        "agent":origin_fe_sub,
        "params":{
            "iocvt_dict": {
                origin_fe_sub.INPUT_image_name:word_tensor_image_name,
                origin_fe_sub.OUTPUT_feature_name:word_feature_name,
            },
            "modcvt_dict": {
                origin_fe_sub.MOD_feature_extractor_name: word_fe_name,
            }
        }
    }

