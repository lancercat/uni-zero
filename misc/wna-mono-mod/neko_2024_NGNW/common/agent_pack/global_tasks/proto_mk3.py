from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent,neko_module_wrapping_agent,neko_parallel_agent_wrapping_agent
from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_sampler_mk2 import get_neko_label_sampler_agent_mk2,get_neko_label_sampler_agent_get_neko_label_sampler_agent_mk2_curriculum
from neko_sdk.neko_framework_NG.agents.prototyping.neko_vis_prototyper_mk3 import neko_vis_prototyper_agent_mk3,neko_vis_prototyper_agent_mk3_rot

class neko_proto_common_mk3:
    MN=mod_names;
    VAN=agent_var_names;

    def get_proto_sampling(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["protosam", "protoenc"],
                "protosam": get_neko_label_sampler_agent_mk2(
                    prefix + this.VAN.RAW_LABEL_NAME,
                    prefix + this.VAN.PROTO_GLOBAL_ID,
                    prefix + this.VAN.GTDICT,
                    prefix + this.VAN.PROTO_LABEL,
                    prefix + this.VAN.TDICT,
                    prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    prefix + this.VAN.PROTO_UTF,
                    this.MN.MVN_name, prefix + this.MN.META_SAM),
                "protoenc": neko_vis_prototyper_agent_mk3.get_agtcfg(
                    prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    prefix + this.VAN.PROTO_VEC,
                    prefix+this.VAN.RAW_PROTO_VEC,
                    prefix+this.MN.PROTO_AGGR, prefix +this.MN.CHAR_ATT,
                    prefix+ this.MN.PROTO_FE)
            }
        }
        return ac;
class neko_proto_common_mk3_rot(neko_proto_common_mk3):
    def __init__(this,possible_rots):
        this.possible_rots=possible_rots;
    def get_proto_sampling(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["protosam", "protoenc"],
                "protosam": get_neko_label_sampler_agent_mk2(
                    prefix + this.VAN.RAW_LABEL_NAME,
                    prefix + this.VAN.PROTO_GLOBAL_ID,
                    prefix + this.VAN.GTDICT,
                    prefix + this.VAN.PROTO_LABEL,
                    prefix + this.VAN.TDICT,
                    prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    prefix + this.VAN.PROTO_UTF,
                    this.MN.MVN_name, prefix + this.MN.META_SAM),
                # It goes this way because we want to keep the trainer till 2025
                "protoenc": neko_vis_prototyper_agent_mk3_rot.get_agtcfg(
                    prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    prefix + this.VAN.PROTO_VEC,
                    prefix + this.VAN.RAW_PROTO_VEC,
                    prefix + this.VAN.PROTO_VEC_ROTATED,
                    prefix + this.VAN.RAW_PROTO_VEC_ROTATED,
                    prefix + this.MN.PROTO_AGGR,
                    prefix + this.MN.CHAR_ATT,
                    prefix + this.MN.PROTO_FE,this.possible_rots)
            }
        }
        return ac;