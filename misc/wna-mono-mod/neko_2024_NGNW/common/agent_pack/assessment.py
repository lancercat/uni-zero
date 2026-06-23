from neko_2024_NGNW.common.namescope import mod_names, agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.data_utils.aug.determinstic_aug_mk2 import get_neko_basic_beacon_agent
from neko_sdk.neko_framework_NG.agents.utils.neko_mvn_agent import get_neko_mvn_agent
from neko_sdk.neko_framework_NG.agents.prototyping.neko_vis_prototyper import neko_vis_prototyper_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf

class neko_beacon_assessment_agent(neko_vis_prototyper_agent):
    OUTPUT_featmap_name="feat_map";
    MOD_fe_name="fe_name";
    MOD_aggr_name="aggr_name";

    def set_etc(this, param):
        this.possible_rotation=[];

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.fe = this.register_mod(this.MOD_fe_name, modcvt_dict);
        this.aggr = this.register_mod(this.MOD_aggr_name, modcvt_dict);

        this.protoimage = this.register(this.INPUT_tensor_proto_img_name, iocvt_dict, this.input_dict);
        this.protovector = this.register(this.OUTPUT_tensor_proto_vec_name, iocvt_dict, this.output_dict);
        this.feat_map = this.register_output(this.OUTPUT_featmap_name, iocvt_dict);

    def make_protos(this,environment,proto_ims):
        feats=environment.module_dict[this.fe](proto_ims);
        protos = environment.module_dict[this.aggr](feats);
        protos = trnf.normalize(protos, p=2, dim=-1);
        return protos,feats;

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        protos,feats = this.make_protos(environment, workspace.inter_dict[this.protoimage]);
        workspace.add(this.protovector,protos);
        workspace.add(this.feat_map,feats[-1]);
        return workspace, environment
def get_beacon_assessment_agent(tensor_proto_img_name,tensor_proto_vec_name,featuremap_name, fe_name,aggr_name):
    return {
        "agent":neko_beacon_assessment_agent,
        "params":{
            "iocvt_dict": {
                neko_beacon_assessment_agent.INPUT_tensor_proto_img_name:tensor_proto_img_name,
                neko_beacon_assessment_agent.OUTPUT_tensor_proto_vec_name:tensor_proto_vec_name,
                neko_beacon_assessment_agent.OUTPUT_featmap_name:featuremap_name,
            },
            "modcvt_dict": {
                neko_beacon_assessment_agent.MOD_fe_name: fe_name,
                neko_beacon_assessment_agent.MOD_aggr_name:aggr_name
            }
        }
    }


class neko_assessment:
    MN = mod_names;
    VAN = agent_var_names;
    BEACON_w = 64;
    BEACON_h = 64;
    def get_beacon_engin(this,prefix):
        return get_neko_basic_beacon_agent(prefix + this.VAN.RAW_IMG_NAME,
                                    prefix + this.VAN.RAW_BEACON_NAME, this.BEACON_h,
                                    this.BEACON_w);
    def get_assessment(this, prefix):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["mkbeacon", "mvn", "assess"],
                "mkbeacon": this.get_beacon_engin(prefix),
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
