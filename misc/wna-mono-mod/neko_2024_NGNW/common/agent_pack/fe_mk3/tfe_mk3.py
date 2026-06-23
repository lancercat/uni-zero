from osocrNG.configs.typical_agent_setups.detached_se_fpn import get_detached_se_fpn,get_se_fpn
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.utils.pool_and_cat import get_neko_pool_and_cat_agent
from neko_sdk.neko_framework_NG.agents.neko_detacher_agent import get_neko_list_detacher_agent
from osocrNG.modular_agents_ocrNG.fe_agents.fpn_like import get_fpn_like_fe
from osocrNG.modular_agents_ocrNG.se_agents.se_agent import transformed_spatial_embedding
# in mk3, tfe is now managed by branch

# no need for temporal info if we only have ctc heads.
class neko_mk3_tfe:
    MN = mod_names;
    VAN = agent_var_names;

    def append_tfe(this,ac,prefix,localprfx="",skip_transformation=False):
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,localprfx+"tfe",
            get_detached_se_fpn(prefix + this.VAN.WORD_FEAT,
                               prefix + this.VAN.WORD_FEAT_DETACHED,
                               prefix + this.VAN.WORD_FEAT_DETACHED_SE,
                               prefix + this.VAN.WORD_TEMP_FEAT,
                               prefix + this.VAN.WORD_TEMP_ENDPOINTS, prefix + this.MN.WORD_TEMPORAL_SE,
                               prefix + this.MN.WORD_TEMPORAL_FE));
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,localprfx+"tfp",
                                                         get_neko_pool_and_cat_agent(
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL));
        return ac;

class neko_mk3_tfe_fse(neko_mk3_tfe):
    def append_tfe(this,ac,prefix,localprfx="",skip_transform=False):

        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfe_detach",
                                                           get_neko_list_detacher_agent([prefix + this.VAN.WORD_FEAT], [
                                                               prefix + this.VAN.WORD_FEAT_DETACHED]));
        # well if the anchor decide that a transform is not necessary, then don't
        if(skip_transform):
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfe",
                                                               get_detached_se_fpn(prefix + this.VAN.WORD_FEAT,
                                                                                   prefix + this.VAN.WORD_FEAT_DETACHED,
                                                                                   prefix + this.VAN.WORD_FEAT_DETACHED_SE,
                                                                                   prefix + this.VAN.WORD_TEMP_FEAT,
                                                                                   prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                                                                                   prefix + this.MN.WORD_TEMPORAL_SE,
                                                                                   prefix + this.MN.WORD_TEMPORAL_FE));
        else:
            fatal("future");

        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfp",
                                                           get_neko_pool_and_cat_agent(
                                                               prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                                                               prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL));
        return ac;


class neko_mk3_tfe_nse(neko_mk3_tfe):
    def append_tfe(this,ac,prefix,localprfx="",skip_transform=False):

        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfe_detach",
                                                           get_neko_list_detacher_agent([prefix + this.VAN.WORD_FEAT], [
                                                               prefix + this.VAN.WORD_FEAT_DETACHED]));
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfe_fpn",
                                                           get_fpn_like_fe(prefix + this.VAN.WORD_FEAT_DETACHED,  prefix + this.VAN.WORD_TEMP_ENDPOINTS, prefix + this.VAN.WORD_TEMP_FEAT, prefix + this.MN.WORD_TEMPORAL_FE));
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, localprfx + "tfp",
                                                           get_neko_pool_and_cat_agent(
                                                               prefix + this.VAN.WORD_TEMP_ENDPOINTS,
                                                               prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL));
        return ac;
