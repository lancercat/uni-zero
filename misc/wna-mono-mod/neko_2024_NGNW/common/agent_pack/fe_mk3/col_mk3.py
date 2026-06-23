
from neko_sdk.neko_framework_NG.agents.utils.neko_mvn_agent import get_neko_mvn_agent

from osocrNG.configs.typical_agent_setups.fe import get_origin_fe


from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.CnC.collate.mask_free_collate_and_packing_agent import neko_mask_free_collate_agent
from neko_sdk.CnC.collate_mk2.grid_making_agent_mk2 import neko_grid_making_agent_mk2
from neko_sdk.neko_framework_NG.agents.utils.padimage import neko_cv2_padding_agent
from neko_sdk.CnC.collate.neko_grid_sample import neko_grid_collate_agent

# in separated collate, a grid maker will always be needed,
# if you don't need a grid, make a dummy one.
# the collate module will always take in a grid, no matter needed or not---
# it's your responsibility to ignore it or use it.
class neko_mk3_collate:
    MN = mod_names;
    VAN = agent_var_names;
    def get_pad_agent(this, prefix):
        return neko_cv2_padding_agent.get_agtcfg(prefix + this.VAN.RAW_IMG_NAME, prefix + this.VAN.RAW_IMG_NAME_PADDED,
                                                 2, 0.05);
    def get_collate_agent(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["pad", "mvn", "mkgrid", "collate"],
                "pad": this.get_pad_agent(prefix),
                "mvn": get_neko_mvn_agent([prefix + this.VAN.RAW_IMG_NAME_PADDED],
                                          [prefix + this.VAN.TEN_IMG_NAME_UNA],
                                          this.MN.IMG_MVN_name),
                "mkgrid": neko_grid_making_agent_mk2.get_agtcfg(
                    prefix + this.VAN.ROUTER_FEATMAP_NAME,
                    prefix + this.VAN.ROUTER_FEAT_NAME,
                    prefix + this.VAN.TEN_IMG_NAME_UNA,
                    prefix + this.VAN.COLLATE_GRID_NAME,
                    prefix + this.MN.COLLATE_GRID_MKER
                ),
                "collate": neko_grid_collate_agent.config_me(
                    [prefix + this.VAN.TEN_IMG_NAME_UNA],
                    prefix + this.VAN.COLLATE_GRID_NAME,
                    [prefix + this.VAN.TEN_IMG_NAME],
                    prefix + this.MN.COLLATE_name)
            }
        }
        return ac

# backport does not pad.
class neko_mk3_collate_backport:
    MN = mod_names;
    VAN = agent_var_names;

    def get_collate_agent(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [ "mvn", "mkgrid", "collate"],
                "mvn": get_neko_mvn_agent([prefix + this.VAN.RAW_IMG_NAME],
                                          [prefix + this.VAN.TEN_IMG_NAME_UNA],
                                          this.MN.IMG_MVN_name),
                "mkgrid": neko_grid_making_agent_mk2.get_agtcfg(
                    prefix + this.VAN.ROUTER_FEATMAP_NAME,
                    prefix + this.VAN.ROUTER_FEAT_NAME,
                    prefix + this.VAN.TEN_IMG_NAME_UNA,
                    prefix + this.VAN.COLLATE_GRID_NAME,
                    prefix + this.MN.COLLATE_GRID_MKER
                ),
                "collate": neko_grid_collate_agent.config_me(
                    [prefix + this.VAN.TEN_IMG_NAME_UNA],
                    prefix + this.VAN.COLLATE_GRID_NAME,
                    [prefix + this.VAN.TEN_IMG_NAME],
                    prefix + this.MN.COLLATE_name)
            }
        }
        return ac

