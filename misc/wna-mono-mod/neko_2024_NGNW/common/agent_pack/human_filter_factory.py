from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from osocrNG.modular_agents_ocrNG.filter_agents.pre_ar_filter_agent import get_neko_pre_aspect_ratio_filter_agent_np
from osocrNG.modular_agents_ocrNG.filter_agents.pre_length_filter_agent import get_neko_pre_length_filter_agent
from osocrNG.modular_agents_ocrNG.filter_agents.pre_closest_ar_filter_agent import get_neko_pre_closest_aspect_ratio_filter_agent_np
from neko_2024_NGNW.common.ak6 import AK6

# some rule based filter:
# it filters off obvious bad solutions, so we don't have to sample
class neko_base_human_filter:
    MN = mod_names;
    VAN = agent_var_names;
    def get_prior_filters_testing(this, prefix, anchor_dict):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                neko_agent_wrapping_agent.PARAM_AGT_LST: ["arfilter"],  # do we want a nofilter here???
                "arfilter": get_neko_pre_aspect_ratio_filter_agent_np(
                    prefix + this.VAN.RAW_IMG_NAME,
                    prefix+this.VAN.ROUTER_MASK_NAME_EMPTY,  # for testing we do not have the length of the gt. dummy.
                    prefix+this.VAN.ROUTER_MASK_NAME, # if this is not final, don't use final names!
                    [anchor_dict[a][AK6.max_routing_ratio] for a in anchor_dict[AK6.names]],
                    [anchor_dict[a][AK6.min_routing_ratio] for a in anchor_dict[AK6.names]]
                )
                # so sleepy...
            }
        }
        return ac;
    def get_prior_filters_training(this, prefix, anchor_dict):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                neko_agent_wrapping_agent.PARAM_AGT_LST: ["lenfilter", "arfilter"],  # do we want a nofilter here???
                "lenfilter": get_neko_pre_length_filter_agent(
                    prefix + this.VAN.RAW_LABEL_NAME,
                    prefix + this.VAN.RAW_IMG_NAME,
                    prefix+this.VAN.ROUTER_MASK_NAME_EMPTY,
                    prefix+this.VAN.ROUTER_MASK_NAME_LEN,
                    [anchor_dict[a][AK6.maxT] for a in anchor_dict[AK6.names]],
                    [anchor_dict[a][AK6.minT] for a in anchor_dict[AK6.names]]
                ),
                "arfilter": get_neko_pre_aspect_ratio_filter_agent_np(
                    prefix + this.VAN.RAW_IMG_NAME,
                    prefix+this.VAN.ROUTER_MASK_NAME_LEN,
                    prefix+this.VAN.ROUTER_MASK_NAME, # if this is not final, don't use final names!
                    [anchor_dict[a][AK6.max_routing_ratio] for a in anchor_dict[AK6.names]],
                    [anchor_dict[a][AK6.min_routing_ratio] for a in anchor_dict[AK6.names]]
                )
                # so sleepy...
            }
        }
        return ac;


# force the router to route the data to actors with closest aspect ratio.
class neko_rule_based_base_human_filter(neko_base_human_filter):
    def get_prior_filters_training(this,prefix,anchor_dict):
        ac={
            "agent":neko_agent_wrapping_agent,
            "params":{
                    neko_agent_wrapping_agent.PARAM_AGT_LST: ["lenfilter","arfilter","closestar"], # do we want a nofilter here???
                    "lenfilter": get_neko_pre_length_filter_agent(
                        prefix+this.VAN.RAW_LABEL_NAME,
                        prefix+this.VAN.RAW_IMG_NAME,
                        prefix+this.VAN.ROUTER_MASK_NAME_EMPTY,
                        prefix+this.VAN.ROUTER_MASK_NAME_LEN,
                        [anchor_dict[a][AK6.maxT] for a in anchor_dict[AK6.names]],
                        [anchor_dict[a][AK6.minT] for a in anchor_dict[AK6.names]]
                    ),
                    "arfilter": get_neko_pre_aspect_ratio_filter_agent_np(
                        prefix + this.VAN.RAW_IMG_NAME,
                        prefix+this.VAN.ROUTER_MASK_NAME_LEN,
                        prefix+this.VAN.ROUTER_MASK_NAME_LEN_AR,
                        [anchor_dict[a][AK6.max_routing_ratio] for a in anchor_dict[AK6.names]],
                        [anchor_dict[a][AK6.min_routing_ratio] for a in anchor_dict[AK6.names]]
                    ),
                "closestar": get_neko_pre_closest_aspect_ratio_filter_agent_np(
                    prefix + this.VAN.RAW_IMG_NAME,
                    prefix+this.VAN.ROUTER_MASK_NAME_LEN_AR,
                    prefix+this.VAN.ROUTER_MASK_NAME,
                    [anchor_dict[a][AK6.target_size_wh] for a in anchor_dict[AK6.names]],
                    1,
                    ) # ONLY routes to the target with closest
                }
        }
        return ac;

    # force the router to route the data to actors with closest aspect ratio.
    def get_prior_filters_testing(this,prefix,anchor_dict):
        ac={
            "agent":neko_agent_wrapping_agent,
            "params":{
                    neko_agent_wrapping_agent.PARAM_AGT_LST: ["arfilter","closestar"], # do we want a nofilter here???

                    "arfilter": get_neko_pre_aspect_ratio_filter_agent_np(
                        prefix + this.VAN.RAW_IMG_NAME,
                        prefix+this.VAN.ROUTER_MASK_NAME_EMPTY,
                        prefix+this.VAN.ROUTER_MASK_NAME_LEN_AR,
                        [anchor_dict[a][AK6.max_routing_ratio] for a in anchor_dict[AK6.names]],
                        [anchor_dict[a][AK6.min_routing_ratio] for a in anchor_dict[AK6.names]]
                    ),
                "closestar": get_neko_pre_closest_aspect_ratio_filter_agent_np(
                    prefix + this.VAN.RAW_IMG_NAME,
                    prefix+this.VAN.ROUTER_MASK_NAME_LEN_AR,
                    prefix+this.VAN.ROUTER_MASK_NAME,
                    [anchor_dict[a][AK6.target_size_wh] for a in anchor_dict[AK6.names]],
                    1,
                    ) # ONLY routes to the target with closest
                }
        }
        return ac;
