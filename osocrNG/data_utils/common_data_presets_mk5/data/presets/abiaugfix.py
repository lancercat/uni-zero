
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from osocrNG.data_utils.common_data_presets_mk5.data.augmenter.abinet_fixed import neko_abinet_aug_factory,neko_no_aug_factory
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_sdk.neko_framework_NG.names import P


class abi_fixed_training(neko_virtual_factory_mk5):
    DATA_PRFX = "dataprfx";
    DATA_PRFX_AUGED_IMG="augmented_image";
    PARAM_SEED = "seed";

    TRAUG=neko_abinet_aug_factory;

    @classmethod
    def arm_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):

        normal_prfx=data_prfxs[cls.DATA_PRFX];
        augmented_img_prfx=data_prfxs[cls.DATA_PRFX_AUGED_IMG];
        acfg=cls.TRAUG.arm_augment(acfg,{
            cls.TRAUG.DATA_PRE_AUG_PRFX:normal_prfx,
            cls.TRAUG.DATA_PRFX:augmented_img_prfx,
        },mod_prfxs,agt_prfxs,params);
        acfg=awa.append_agent_to_cfg(acfg,P(agt_prfxs[cls.AGT_PRFX],"symlink_label"),neko_symbol_link_alot_agent.get_agtcfg(
            [VN.UTF(normal_prfx),VN.GT_TOK_UTF(normal_prfx),VN.SAM_ORIG(normal_prfx)],
            [VN.UTF(augmented_img_prfx),VN.GT_TOK_UTF(augmented_img_prfx),VN.SAM_ORIG(augmented_img_prfx)]
        )); # now augment create its own data stream. Better consistance and less technical debt
        return acfg;

# this has to be reworked--- everything geometry should now be grid based.
class noaug_training(abi_fixed_training):

    TRAUG=neko_no_aug_factory;
