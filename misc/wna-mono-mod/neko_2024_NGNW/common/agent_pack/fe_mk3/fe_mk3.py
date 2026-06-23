
from neko_sdk.neko_framework_NG.agents.utils.neko_mvn_agent import get_neko_mvn_agent

from osocrNG.configs.typical_agent_setups.fe import get_origin_fe


from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.CnC.collate.mask_free_collate_and_packing_agent import neko_mask_free_collate_agent
from neko_sdk.CnC.collate.neko_grid_sample import get_neko_grid_making_agent,neko_grid_collate_agent,neko_static_grid_mask_free_collate_agent
from neko_sdk.neko_framework_NG.agents.utils.padimage import neko_cv2_padding_agent

class neko_mk3_fe:
    MN = mod_names;
    VAN = agent_var_names;


    def get_fe(this, prefix):
        return get_origin_fe(prefix + this.VAN.TEN_IMG_NAME,
                                    prefix + this.VAN.WORD_FEAT, prefix + this.MN.WORD_FE);

    ## common process of conquerer

