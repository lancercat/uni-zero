
from neko_sdk.neko_framework_NG.agents.utils.neko_mvn_agent import get_neko_mvn_agent

from osocrNG.configs.typical_agent_setups.fe import get_origin_fe


from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.CnC.collate.mask_free_collate_and_packing_agent import neko_mask_free_collate_agent
from neko_sdk.neko_framework_NG.agents.utils.padimage import neko_cv2_padding_agent
# in mk3, tfe is now managed by branch
# no need for temporal info if we only have ctc heads.
#
# class neko_mk6_collate_fe:
#     MN = mod_names;
#     VAN = agent_var_names;
#     def get_collate_agent(this, prefix):
#         ac = {
#             "agent": neko_agent_wrapping_agent,
#             "params": {
#                 "agent_list": ["collate", "mvn"],
#                 "collate":neko_mask_free_collate_agent.config_me(
#             [prefix + this.VAN.RAW_IMG_NAME],
#             [prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
#             prefix + this.MN.COLLATE_name),
#                 "mvn": get_neko_mvn_agent([prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
#                                           [prefix + this.VAN.TEN_IMG_NAME],
#                                           this.MN.MVN_name)
#             }
#         }
#         return ac
#
#     def get_fe(this, prefix):
#         return get_origin_fe(prefix + this.VAN.TEN_IMG_NAME,
#                                     prefix + this.VAN.WORD_FEAT, prefix + this.MN.WORD_FE);
#
#     ## common process of conquerer
#
#     def get_fp_collate_core(this, prefix):
#         return {
#             "agent": neko_agent_wrapping_agent,
#             "params": {
#                 "agent_list": ["collate", "fe"],
#                 "collate": this.get_collate_agent(prefix),
#                 "fe": this.get_fe(prefix)
#             }
#         }
class neko_mk3_collate_fe:
    MN = mod_names;
    VAN = agent_var_names;

    def get_collate_agent(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["collate", "mvn"],
                "collate":neko_mask_free_collate_agent.config_me(
            [prefix + this.VAN.RAW_IMG_NAME],
            [prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
            prefix + this.MN.COLLATE_name),
                "mvn": get_neko_mvn_agent([prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
                                          [prefix + this.VAN.TEN_IMG_NAME],
                                          this.MN.MVN_name)
            }
        }
        return ac

    def get_fe(this, prefix):
        return get_origin_fe(prefix + this.VAN.TEN_IMG_NAME,
                                    prefix + this.VAN.WORD_FEAT, prefix + this.MN.WORD_FE);

    ## common process of conquerer

    def get_fp_collate_core(this, prefix,skip_transform=False): # only used in collate with cmdcol instances.
        return {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["collate", "fe"],
                "collate": this.get_collate_agent(prefix),
                "fe": this.get_fe(prefix),
            }
        }
class neko_mk3_collate_fe_static_grid(neko_mk3_collate_fe):
    MN = mod_names;
    VAN = agent_var_names;

    def get_collate_agent(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [ "mvn","collate"],
                "mvn": get_neko_mvn_agent([prefix + this.VAN.RAW_IMG_NAME],
                                          [prefix + this.VAN.TEN_IMG_NAME_UNA],
                                          this.MN.IMG_MVN_name),
                "collate":neko_static_grid_mask_free_collate_agent.config_me(
            [prefix + this.VAN.TEN_IMG_NAME_UNA],
            [prefix + this.VAN.TEN_IMG_NAME],
            prefix + this.MN.COLLATE_name)
            }
        }
        return ac



class neko_mk3_pad_collate_fe(neko_mk3_collate_fe):
    def get_pad_agent(this,prefix):
        return neko_cv2_padding_agent.get_agtcfg(prefix+this.VAN.RAW_IMG_NAME,prefix+this.VAN.RAW_IMG_NAME_PADDED,2,0.05);
    def get_collate_agent(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["pad","collate", "mvn"],
                "pad": this.get_pad_agent(prefix),
                "collate":neko_mask_free_collate_agent.config_me(
            [prefix + this.VAN.RAW_IMG_NAME_PADDED],
            [prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
            prefix + this.MN.COLLATE_name),
                "mvn": get_neko_mvn_agent([prefix + this.VAN.ALIGNED_RAW_IMG_NAME],
                                          [prefix + this.VAN.TEN_IMG_NAME],
                                          this.MN.MVN_name)
            }
        }
        return ac



