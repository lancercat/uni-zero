# we try to setup everything but data in one factory.
# I have 1000 line files but let's keep this way,
# before I figure out what part can be decoupled from this mess.
from typing import Dict

from neko_sdk.neko_framework_NG.UAE.neko_trainer_agent import neko_trainer_agent

from neko_sdk.CnC.aggregate.simple_agent_aggr import get_neko_max_conf_aggr_agent
from neko_sdk.neko_framework_NG.UAE.neko_mission_agent import neko_test_mission_agent,neko_test_mission_agent_single_im
from neko_sdk.neko_framework_NG.agents.debugging.workspace_dumper import get_neko_workspace_dumper

from osocrNG.modular_agents_ocrNG.output_logging_subs.basic_acr_fps_reporter import case_inv_acr_fps_reporter

from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.result_rendering import get_neko_result_rendering_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.attention_visualization import \
    get_attention_visualization_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.matching_flair_renders import \
    get_neko_perfect_match_flair_making_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.tensor_im_visualization import \
    get_neko_tensor_im_visualization
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.remix_agent import get_neko_remix_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.routing_flair_renders import \
    get_neko_routing_flair_making_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.log_saver.simple_log_saver import get_ocr_simple_logging_agent
# we try to setup everything but data in one factory.
# I have 1000 line files but let's keep this way,
# before I figure out what part can be decoupled from this mess.
from neko_2024_NGNW.common.namescope import mod_names, agent_var_names

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.configs.typical_agent_setups.ocr_agents.visocr_agent import get_show_batch_agent
class debugger:

    MN=mod_names;
    VAN=agent_var_names;
    def get_reporters_core(this, prefix,pred_text_name_dict):
        rd = {};
        a = case_inv_acr_fps_reporter;
        for target_name in pred_text_name_dict:
            rd[prefix +pred_text_name_dict[target_name]["variant"] + "case_inv_acr_time"] = {
                "agent": a,
                "params": {
                    a.PARAM_variant: pred_text_name_dict[target_name]["variant"],
                    "iocvt_dict": {
                        a.INPUT_tdict_name: prefix + this.VAN.TDICT,
                        a.INPUT_pred_text_name: target_name,
                        a.INPUT_raw_label_name: prefix + this.VAN.RAW_LABEL_NAME
                    },
                    "modcvt_dict": {},
                }
            }
        return rd;
    def get_reporters(this, prefix):
        return this.get_reporters_core(prefix,{prefix+this.VAN.PRED_TEXT:{"variant":""}});
    def get_dbg_trainer(this, val_dict: Dict, anchor, qname, ecnt=5, icnt=200000):
        trainer_param = this.get_trainer_param(val_dict, anchor, qname, ecnt, icnt);
        for n in trainer_param["routine_names"]:
            for aname in anchor["names"]:
                trainer_param["routine_dict"][n]["params"]["agent_list"].append(aname + "debug_visualize");
                trainer_param["routine_dict"][n]["params"][aname + "debug_visualize"] = get_show_batch_agent(aname,
                                                                                                             aname + this.VAN.TEN_IMG_NAME,
                                                                                                             aname + this.VAN.RAW_LABEL_NAME);
        e = neko_trainer_agent;
        print(trainer_param)
        trainer = e(
            trainer_param
        )
        return trainer;

    def get_aggr_agent(this, target, prefixes, gprfx=""):
        # return get_neko_simple_aggr_agent(
        #     [gprfx + p + this.VAN.ROUTER_SAM_ID  for p in prefixes],
        #     [gprfx+p+target for p in prefixes],
        #     this.VAN.RAW_IMG_NAME,
        #     target,
        #     "NEP_skipped_NEP",prefixes
        # );
        return get_neko_max_conf_aggr_agent(
            [gprfx + p + this.VAN.ROUTER_SAM_ID for p in prefixes],
            [gprfx + p + this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME for p in prefixes],
            [gprfx + p + target for p in prefixes],
            target,
            "NEP_skipped_NEP"
        );


    def get_log_keys(this, anchor_dict):
        return None;

    def append_core_dump_agent(this, agent_cfg, anchor_dict, logpath, saveprfx, prefix=""):
        an = prefix + "coredump";
        sac = get_neko_workspace_dumper(prefix + this.VAN.RAW_IMG_TAG, this.get_log_keys(anchor_dict["names"]), logpath,
                                        saveprfx);
        agent_cfg = neko_agent_wrapping_agent.append_agent_to_cfg(agent_cfg, an, sac);
        return agent_cfg;


    # has_gt--do we need to compare gt and prediction visually?
    def get_result_vis(this, prefix, gprfix,has_gt=False):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["visatt", "vis_tensor_im", "correctness_flair", "routing_flair", "render", "remix"],
                "visatt": get_attention_visualization_agent(prefix +"main_"+ this.VAN.ATT_MAP, prefix + this.VAN.TEN_IMG_NAME,
                                                            prefix+"main_"+this.VAN.ATT_LEN_PRED_AMAX, prefix + this.VAN.DBG_ATT_IMS),
                "vis_tensor_im": get_neko_tensor_im_visualization(prefix + this.VAN.TEN_IMG_NAME,
                                                                  prefix + this.VAN.DBG_PADDED_RAW_IM, 127, 127),
                "correctness_flair": get_neko_perfect_match_flair_making_agent(
                    prefix + this.VAN.PRED_TEXT, prefix + this.VAN.RAW_LABEL_NAME,
                    prefix + this.VAN.DBG_MATCHING_FLAIRS),
                "routing_flair": get_neko_routing_flair_making_agent(gprfix+prefix+ this.VAN.ROUTER_ACT_NAME,
                                                                     prefix + this.VAN.DBG_ROUTING_FLAIRS,
                                                                     "NEP_default_NEP"),
                "render": get_neko_result_rendering_agent(
                    gprfix + this.VAN.PROTO_LABEL, prefix + this.VAN.PRED_TEXT,
                    gprfix + this.VAN.TDICT, gprfix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    prefix + this.VAN.RAW_LABEL_NAME, prefix + this.VAN.DBG_GT_VPR_PATCHES,
                    "NEP_default_NEP"),
                "remix": get_neko_remix_agent(
                    [prefix + this.VAN.DBG_MATCHING_FLAIRS, prefix + this.VAN.DBG_ROUTING_FLAIRS],
                    [prefix + this.VAN.DBG_GT_VPR_PATCHES],
                    [prefix + this.VAN.DBG_PADDED_RAW_IM,
                     prefix + this.VAN.DBG_ATT_IMS], prefix + this.VAN.RAW_IMG_NAME,
                                                     prefix + this.VAN.DBG_RESULT_PANEL),
                neko_agent_wrapping_agent.PARAM_ACT_VARS: [prefix + this.VAN.RAW_IMG_NAME]
            },
        }
        return ac;

    def append_result_loggers(this, agent_cfg, anchor_dict, logpath, saveprfx, prefix=""):
        # well each anchor will haz its own visualizer.
        # We will find a way to aggregate the visualized results.....
        # inject extra broadcast item.
        agent_cfg["params"]["command"]["params"]["control"]["params"]["routing_subjects"].append(prefix+this.VAN.ROUTER_ACT_NAME)
        for a in anchor_dict["names"]:
            lprfx = prefix + a;
            tag = lprfx + "visualizer";
            sac = this.get_result_vis(lprfx, prefix);
            agent_cfg = neko_agent_wrapping_agent.append_agent_to_cfg(agent_cfg, tag, sac);
        return agent_cfg;

    def as_debugger(this,cta, benchdict, anchor_dict, logpath, saveprfx, prefix="", core_dump=False, forcek=1):
        cta = this.append_result_loggers(cta, anchor_dict, logpath, saveprfx, prefix);
        if (core_dump):
            cta = this.append_core_dump_agent(cta, anchor_dict, logpath, saveprfx, prefix);
        sac = this.get_aggr_agent(prefix + this.VAN.DBG_RESULT_PANEL, anchor_dict["names"], prefix);
        neko_agent_wrapping_agent.append_agent_to_cfg(cta, "vis_aggr", sac);
        sac = get_ocr_simple_logging_agent(prefix + this.VAN.RAW_IMG_TAG,
                                           prefix + this.VAN.RAW_LABEL_NAME,
                                           [prefix + this.VAN.DBG_RESULT_PANEL],
                                           prefix + this.VAN.PRED_TEXT, logpath, ["id"]);
        neko_agent_wrapping_agent.append_agent_to_cfg(cta, "vis_dmp", sac);
        for a in anchor_dict["names"]:
            sac = get_ocr_simple_logging_agent(prefix + a + this.VAN.RAW_IMG_TAG,
                                               prefix + a + this.VAN.RAW_LABEL_NAME,
                                               [prefix + a + this.VAN.DBG_RESULT_PANEL],
                                               prefix + a + this.VAN.PRED_TEXT, logpath, ["id"], prefix + a);
            neko_agent_wrapping_agent.append_agent_to_cfg(cta, prefix + a + "vis_dmp", sac);
        ta = neko_test_mission_agent;
        return {
            "agent": ta,
            "params": {
                ta.PARAM_test_benches: benchdict,
                ta.AGENT_reporter_dict: this.get_reporters(prefix),
                ta.PARAM_test_variant: "",
                ta.AGENT_tester_agent: cta,
                "iocvt_dict": {
                    ta.OUTPUT_tdict_name: prefix + this.VAN.TDICT,
                    ta.OUTPUT_raw_id_name: prefix + this.VAN.RAW_IMG_TAG,
                    ta.OUTPUT_raw_label_name: prefix + this.VAN.RAW_LABEL_NAME,
                    ta.OUTPUT_weightroto_label_name: prefix + this.VAN.PROTO_LABEL,
                    ta.OUTPUT_tensor_proto_img_name: prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    ta.OUTPUT_raw_image_name: prefix + this.VAN.RAW_IMG_NAME,
                    ta.OUTPUT_tensor_proto_vec_name: prefix + this.VAN.PROTO_VEC
                },
                "modcvt_dict": {
                    ta.MOD_proto_mvn_name: this.MN.MVN_name,
                    ta.MOD_prototyper_name: this.MN.PROTO_ENC
                }

            }
        };
    def as_athena(this,cta, benchdict, anchor_dict, logpath, saveprfx, prefix="", core_dump=False, forcek=1):
        cta = this.append_result_loggers(cta, anchor_dict, logpath, saveprfx, prefix);
        if (core_dump):
            cta = this.append_core_dump_agent(cta, anchor_dict, logpath, saveprfx, prefix);
        sac = this.get_aggr_agent(prefix + this.VAN.DBG_RESULT_PANEL, anchor_dict["names"], prefix);
        neko_agent_wrapping_agent.append_agent_to_cfg(cta, "vis_aggr", sac);
        sac = get_ocr_simple_logging_agent(prefix + this.VAN.RAW_IMG_TAG,
                                           prefix + this.VAN.RAW_LABEL_NAME,
                                           [prefix + this.VAN.DBG_RESULT_PANEL],
                                           prefix + this.VAN.PRED_TEXT, logpath, ["id"]);
        neko_agent_wrapping_agent.append_agent_to_cfg(cta, "vis_dmp", sac);
        for a in anchor_dict["names"]:
            sac = get_ocr_simple_logging_agent(prefix + a + this.VAN.RAW_IMG_TAG,
                                               prefix + a + this.VAN.RAW_LABEL_NAME,
                                               [prefix + a + this.VAN.DBG_RESULT_PANEL],
                                               prefix + a + this.VAN.PRED_TEXT, logpath, ["id"], prefix + a);
            neko_agent_wrapping_agent.append_agent_to_cfg(cta, prefix + a + "vis_dmp", sac);
        ta = neko_test_mission_agent_single_im;
        return {
            "agent": ta,
            "params": {
                ta.PARAM_test_benches: benchdict,
                ta.AGENT_reporter_dict: this.get_reporters(prefix),
                ta.PARAM_test_variant: "",
                ta.AGENT_tester_agent: cta,
                "iocvt_dict": {
                    ta.OUTPUT_tdict_name: prefix + this.VAN.TDICT,
                    ta.OUTPUT_raw_id_name: prefix + this.VAN.RAW_IMG_TAG,
                    ta.OUTPUT_raw_label_name: prefix + this.VAN.RAW_LABEL_NAME,
                    ta.OUTPUT_weightroto_label_name: prefix + this.VAN.PROTO_LABEL,
                    ta.OUTPUT_tensor_proto_img_name: prefix + this.VAN.TENSOR_PROTO_IMG_NAME,
                    ta.OUTPUT_raw_image_name: prefix + this.VAN.RAW_IMG_NAME,
                    ta.OUTPUT_tensor_proto_vec_name: prefix + this.VAN.PROTO_VEC
                },
                "modcvt_dict": {
                    ta.MOD_proto_mvn_name: this.MN.MVN_name,
                    ta.MOD_prototyper_name: this.MN.PROTO_ENC
                }
            }
        };