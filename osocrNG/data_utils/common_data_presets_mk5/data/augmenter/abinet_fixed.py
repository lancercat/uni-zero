import cv2

from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from osocrNG.data_utils.aug.abisync import neko_fixed_abinet_aug_agent
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_data_factory_mk5
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.names import P

from neko_sdk.cfgtool.argsparse import neko_get_arg
# this parasites the main data stream
# forking other info is out of scope of augment agents.
class neko_abinet_aug_factory(neko_virtual_data_factory_mk5):
    DATA_PRE_AUG_PRFX="pre_aug_data_prfx";
    @classmethod
    def arm_augment(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        preaugprfx=data_prfxs[cls.DATA_PRE_AUG_PRFX];
        data_prefix = data_prfxs[cls.DATA_PRFX];

        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        acfg=awa.append_agent_to_cfg(acfg,
            P(aprfx,"augment"),neko_fixed_abinet_aug_agent.get_agtcfg(
                VN.IM_raw(preaugprfx),VN.IM_raw(data_prefix),cv2.BORDER_REPLICATE,9
            ));
        return acfg;
class neko_no_aug_factory(neko_virtual_data_factory_mk5):
    DATA_PRE_AUG_PRFX="pre_aug_data_prfx";
    @classmethod
    def arm_augment(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        preaugprfx=data_prfxs[cls.DATA_PRE_AUG_PRFX];
        data_prefix = data_prfxs[cls.DATA_PRFX];

        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        acfg=awa.append_agent_to_cfg(acfg,
            P(aprfx,"augment"),neko_symbol_link_alot_agent.get_agtcfg(
                [VN.IM_raw(preaugprfx)],[VN.IM_raw(data_prefix)]
            ));
        return acfg;
