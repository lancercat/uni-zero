from neko_2024_NGNW.common.agent_pack.cnc_mk3.assessment_mk3 import neko_assessment_mk3
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.CnC.command.agents.policy_sampling.sample_best import get_neko_bestk_policy_agent
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
import copy;
from neko_sdk.CnC.controls.agents.routers.single_step_router import get_neko_single_step_name_based_routing_agent_static
from neko_sdk.neko_framework_NG.agents.massage_passing.neko_broadcasting_agent_static import get_neko_broadcasting_agent_static_single_dev_just_assign
from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_label_making_agent import get_neko_label_making_agent

class neko_cnc_mk3:
    MN = mod_names;
    VAN = agent_var_names;

    PARAM_beacon_h="beacon_h";
    PARAM_beacon_w="beacon_w";

    def set_assessment(this):
        this.assessment_factory=neko_assessment_mk3();

    def __init__(this):
        this.set_assessment();
