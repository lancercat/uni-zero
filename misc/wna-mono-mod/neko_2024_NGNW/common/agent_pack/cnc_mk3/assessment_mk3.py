from neko_2024_NGNW.common.namescope import mod_names, agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.data_utils.aug.determinstic_aug_mk2 import get_neko_basic_beacon_agent,get_neko_grid_beacon_agent,get_neko_gray_grid_beacon_agent
from neko_sdk.neko_framework_NG.agents.utils.neko_mvn_agent import get_neko_mvn_agent
from torch.nn import functional as trnf
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2024_NGNW.common.agent_pack.assessment import get_beacon_assessment_agent
from neko_2024_NGNW.common.ak6 import AK6



class neko_assessment_mk3:
    MN = mod_names;
    VAN = agent_var_names;
    DFT_size_hw=(64,64);


    def get_beacon_engin(this,prefix,bh,bw):
        return get_neko_basic_beacon_agent(prefix + this.VAN.RAW_IMG_NAME,
                                    prefix + this.VAN.RAW_BEACON_NAME, bh,
                                    bw);
    def get_assessment(this, prefix,anchor_dict):
        (bw,bh)=neko_get_arg(AK6.beacon_size_wh, anchor_dict,this.DFT_size_hw);
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["mkbeacon", "mvn", "assess"],
                "mkbeacon": this.get_beacon_engin(prefix,bh,bw),
                "mvn": get_neko_mvn_agent([prefix + this.VAN.RAW_BEACON_NAME],
                                          [prefix + this.VAN.TENSOR_BEACON_NAME],
                                          this.MN.MVN_name),
                "assess": get_beacon_assessment_agent(prefix + this.VAN.TENSOR_BEACON_NAME,
                                                prefix + this.VAN.ROUTER_FEAT_NAME,
                                                prefix + this.VAN.ROUTER_FEATMAP_NAME,
                                                prefix + this.MN.ROUTER_FE_bbn_name,
                                                prefix+this.MN.ROUTER_AGGR_name)
            }
        }
        return ac;
class neko_no_assessment_mk3(neko_assessment_mk3):
    def get_assessment(this, prefix,anchor_dict):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [],
            }};
        return ac;


class neko_assessment_mk3_g4x4(neko_assessment_mk3):
    hpartz=4;
    wpartz=4;
    def get_beacon_engin(this,prefix,bh,bw):
        return get_neko_gray_grid_beacon_agent(prefix + this.VAN.RAW_IMG_NAME,
                                    prefix + this.VAN.RAW_BEACON_NAME, bh,
                                    bw,this.hpartz,this.wpartz);
