# we try to setup everything but data in one factory.
# I have 1000 line files but let's keep this way,
# before I figure out what part can be decoupled from this mess.
import copy
import json
from typing import Dict
from neko_sdk.neko_framework_NG.agents.neko_loss_backward_all_agent import get_neko_basic_backward_all_agent
from neko_sdk.neko_framework_NG.agents.loss_logging_agent import get_neko_logging_agent
from neko_sdk.neko_framework_NG.UAE.neko_trainer_agent import neko_trainer_agent
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.UAE.neko_mission_agent import neko_test_mission_agent,neko_test_mission_agent_single_im
from neko_2024_NGNW.common.ak6 import AK6
from osocrNG.modular_agents_ocrNG.pred_subs.simple_pred_dict import get_translate_gt_agent


from osocrNG.modular_agents_ocrNG.output_logging_subs.basic_acr_fps_reporter import case_inv_acr_fps_reporter

from neko_2024_NGNW.common.namescope import mod_names, agent_var_names

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.CnC.aggregate.simple_agent_aggr import get_neko_max_conf_aggr_agent

from neko_2024_NGNW.common.agent_pack.training_mk3 import neko_training_common_mk3
from neko_2024_NGNW.common.agent_pack.global_tasks.proto_mk3 import neko_proto_common_mk3,neko_proto_common_mk3_rot
from neko_2024_NGNW.common.agent_pack.branches_mk3.branch_mk3 import neko_branch_mk3_single_head
from neko_2024_NGNW.common.agent_pack.global_tasks.base_mk3 import neko_base_global_tasks_mk3
from neko_2024_NGNW.common.agent_pack.human_filter_factory import neko_base_human_filter
# from neko_2024_NGNW.common.agent_pack.global_tasks.lsct.basic_lsct_char import
from neko_2024_NGNW.common.agent_pack.debugger.basedebugger import debugger
from neko_2024_NGNW.common.agent_pack.cnc_mk3.assessment_mk3 import neko_assessment_mk3
from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_label_making_agent import get_neko_label_making_agent
from neko_sdk.CnC.command.agents.policy_sampling.sample_best import get_neko_bestk_policy_agent

from neko_sdk.CnC.controls.agents.routers.single_step_router import get_neko_single_step_name_based_routing_agent_static
from neko_sdk.neko_framework_NG.agents.massage_passing.neko_broadcasting_agent_static import get_neko_broadcasting_agent_static_single_dev_just_assign
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent as awa

# from v6 on, not all heads will be covered by at least one expert for each sample.
# its your resposibility, to design fair feasibility measurement and not compare apple (CTC loss) to orange (Xent).

class abstract_agent_factory_v6:
    MN = mod_names;
    VAN = agent_var_names;

    WF_Drop_MN=None;
    CF_Drop_MN=None;

    SAMPLE_k = 9999;
    AGT_W_delay_e = 0;
    # we will probably do not support drop proto till v6.
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head();
    def set_global_task_factory(this):
        this.global_task_factory=neko_base_global_tasks_mk3();


    def extra_routing_subjects(this,prefix):
        return [];
    def set_debugging_engine(this):
        this.dbge = debugger();

    def set_assessment_factory(this):
        this.ass_factory=neko_assessment_mk3();
    def set_hfc_factory(this):
        this.hfc_factory=neko_base_human_filter(); # sample less, sample smart :)
    def set_proto_factory(this):
        this.proto_factory=neko_proto_common_mk3();
    def set_engines(this):
        this.training_factory=neko_training_common_mk3();
        this.set_proto_factory();
        this.set_branch_factory();
        this.set_assessment_factory();
        this.set_global_task_factory();
        this.set_debugging_engine();
        this.set_hfc_factory();


    def setvd(this):
        this.vd={};

    def make_cnc(this, prefix, anchor_dict, hfc, gtc, command, conquer,assc):
        agtlst = ["assessment"];
        params = {
            "assessment": assc,
        };
        if (hfc is not None):
            agtlst.append("human_filters");
            params["human_filters"] = hfc;
        if (gtc is not None):
            agtlst.append("global_tasks");
            params["global_tasks"] = gtc;
        agtlst.append("command");
        params["command"] = command;
        agtlst.append("conquer");
        params["conquer"] = conquer;
        params["agent_list"] = agtlst;
        return {
            "agent": neko_agent_wrapping_agent,
            "params": params
        };

    def __init__(this, platformcfg: neko_platform_cfg):
        this.platform = platformcfg;
        this.set_engines();
        this.setvd();

    def get_conqueror_training(this, prefix, anchor_dict):
        aac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": copy.deepcopy(anchor_dict["names"])
            }
        }
        for a in anchor_dict["names"]:
            aac["params"][a] = this.branch_factory.get_training_fp_branch_agent(prefix,a,anchor_dict[a]);
        return aac;

    def get_prefixed_training_fp_core(this, prefix, anchor_dict):
        gtc = awa.empty();
        gtc, routing_subjects, broadcasting_subjects = this.global_task_factory.append_global_task(gtc,prefix, anchor_dict);

        routing_subjects+=this.extra_routing_subjects(prefix);

        gtc = neko_agent_wrapping_agent.append_agent_to_cfg(gtc, "prototyping", this.proto_factory.get_proto_sampling(prefix));
        # only need to broadcast during training
        broadcasting_subjects += [ prefix + this.VAN.GTDICT,
                                  prefix + this.VAN.PROTO_GLOBAL_ID,
                                  prefix + this.VAN.TENSOR_PROTO_IMG_NAME
                                  ];
        # after sampling prototypes, we can now filter gt text--- if we need ned as a part of reward function---
        # this is better than compute this at head--- we can well forget to, and it comes along sampler afterall
        neko_agent_wrapping_agent.append_agent_to_cfg(gtc, "set_unk_to_gt",
                                get_translate_gt_agent(prefix + this.VAN.RAW_LABEL_NAME, prefix + this.VAN.TDICT,
                                                       this.VAN.RAW_WUNK_LABEL_NAME));

        # only hack this during debugging inference to avoid leaking.
        routing_subjects += [prefix + this.VAN.RAW_WUNK_LABEL_NAME, prefix + this.VAN.RAW_LABEL_NAME]; # training only

        cac = this.get_commander_training(prefix, anchor_dict, routing_subjects=routing_subjects,
                                          broadcasting_subjects=broadcasting_subjects);
        aac = this.get_conqueror_training(prefix, anchor_dict);
        hfc = this.hfc_factory.get_prior_filters_training(prefix, anchor_dict);
        assc=this.ass_factory.get_assessment(prefix,anchor_dict);
        cnc_vm = this.make_cnc(prefix, anchor_dict, hfc, gtc, cac, aac,assc);

        return cnc_vm;
    # this api will be rethought and refactored in v7. Right now v6
    def get_training_agent(this, anchor_dict, queue_name, prefixes=None):
        assert prefixes == None;
        ac = this.get_prefixed_training_fp_core("", anchor_dict);
        ac= neko_agent_wrapping_agent.prepend_agent_to_cfg(ac, "ferrier", this.training_factory.get_data_ferrier(queue_name));

        ac = this.global_task_factory.arm_global_loss(ac, anchor_dict, "");
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
                                                           this.VAN.ANAME_backward_agent,
                                                           get_neko_basic_backward_all_agent());
        return ac;



    def get_commander_training(this, prefix, anchor_dict, routing_subjects, broadcasting_subjects):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["mkgt","command", "control", "broadcast"],
                "mkgt":get_neko_label_making_agent(
                    prefix  + this.VAN.PROTO_VEC,
                    prefix  + this.VAN.RAW_LABEL_NAME,
                    prefix  + this.VAN.TDICT,
                    prefix  + this.VAN.TEN_GT_LEN,
                    prefix  + this.VAN.TENSOR_LABEL_NAME,
                ),
                "command": get_neko_bestk_policy_agent(
                    prefix + this.VAN.ROUTER_MASK_NAME,
                    prefix + this.VAN.ROUTER_FEAT_NAME,
                    prefix + this.VAN.ROUTER_ACT_NAME,
                    prefix + this.VAN.ROUTER_LOGIT,
                    prefix + this.VAN.ROUTER_LOGPROB, this.MN.ROUTER_CMD_name, 9999),
                "control": get_neko_single_step_name_based_routing_agent_static(
                    prefix + this.VAN.ROUTER_ACT_NAME,
                    prefix + this.VAN.ROUTER_LOGPROB,
                    prefix + this.VAN.ROUTER_PATH_LOG_PROB_NAME,
                    this.VAN.ROUTER_PATH_ID_NAME,
                    this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME,
                    prefix + this.VAN.ROUTER_SAM_ID,
                    copy.deepcopy(anchor_dict["names"]),
                    prefix,
                    [prefix + i for i in routing_subjects]
                ),

                "broadcast": get_neko_broadcasting_agent_static_single_dev_just_assign(
                    prefix,
                    [prefix + i for i in broadcasting_subjects],
                    copy.deepcopy(anchor_dict["names"]),
                )
            }
        }
        # Trivia: static means you cannot drop-in new agents(conquerors)
        # in llama-heart-one(2025/26), we will make it work in a way like LLM extensions
        # too many to do, too few time to code. It will be nice if I can fork myself like ninjas.
        return ac;
    def get_commander_testing(this, prefix, anchor_dict, routing_subjects, broadcasting_subjects, k=1):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["command", "control", "broadcast"],
                "command": get_neko_bestk_policy_agent(
                    prefix + this.VAN.ROUTER_MASK_NAME,
                    prefix + this.VAN.ROUTER_FEAT_NAME,
                    prefix + this.VAN.ROUTER_ACT_NAME,
                    prefix + this.VAN.ROUTER_LOGIT,
                    prefix + this.VAN.ROUTER_LOGPROB, this.MN.ROUTER_CMD_name, k),
                "control": get_neko_single_step_name_based_routing_agent_static(
                    prefix + this.VAN.ROUTER_ACT_NAME,
                    prefix + this.VAN.ROUTER_LOGPROB,
                    prefix + this.VAN.ROUTER_PATH_LOG_PROB_NAME,
                    this.VAN.ROUTER_PATH_ID_NAME,
                    this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME,
                    prefix + this.VAN.ROUTER_SAM_ID,
                    copy.deepcopy(anchor_dict["names"]),
                    prefix,
                    [prefix + i for i in routing_subjects]
                ),
                "broadcast": get_neko_broadcasting_agent_static_single_dev_just_assign(
                    prefix,
                    [prefix + i for i in broadcasting_subjects],
                    copy.deepcopy(anchor_dict["names"]),
                )
            }
        }
        # Trivia: static means you cannot drop-in new agents(conquerors)
        # in llama-heart-one(2025/26), we will make it work in a way like LLM extensions
        # too many to do, too few time to code. It will be nice if I can fork myself like ninjas.
        return ac;



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
    def get_testing_fp_agent(this, prefix, anchor_dict, k=1):
        gtc=awa.empty();
        gtc, routing_subjects, broadcasting_subjects = this.global_task_factory.append_global_task(gtc,prefix, anchor_dict);
        routing_subjects+=this.extra_routing_subjects(prefix)+[prefix+this.VAN.RAW_LABEL_NAME]; # so that the debugger logs diff.
        cac = this.get_commander_testing(prefix, anchor_dict, routing_subjects, broadcasting_subjects, k);
        hfc = this.hfc_factory.get_prior_filters_testing(prefix, anchor_dict);
        aac = this.get_conqueror_testing(prefix, anchor_dict);
        assc=this.ass_factory.get_assessment(prefix,anchor_dict);
        cnc_vm=this.make_cnc(prefix, anchor_dict, hfc, gtc, cac, aac,assc);
        return cnc_vm;
    def get_conqueror_testing(this, prefix, anchor_dict,gprfx=""):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": copy.deepcopy(anchor_dict[AK6.names])
            }
        }
        for a in anchor_dict[AK6.names]:
            ac["params"][a] = this.branch_factory.get_testing_fp_branch_agent(prefix,a,anchor_dict[a]);
        # from mk3 on, not all heads will be covered by at least one expert for each sample.
        # You can get no per-head monitor by default,
        # and it will be YOUR resposibility to handle that.
        an = this.VAN.PRED_TEXT + "_" + this.VAN.ANAME_aggr;
        sac = this.get_aggr_agent(this.VAN.PRED_TEXT, anchor_dict["names"]);
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, an, sac);
        for k in this.vd: # in a multi-head setup, you may want to monitor individual heads instead of just the aggregated result.
            an = k+this.VAN.ANAME_aggr;
            sac = this.get_aggr_agent(this.vd[k]["pred"], anchor_dict["names"],gprfx);
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, an, sac)
        return ac;

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


    # I mean you may want your logger your way. Inherient and hack this.
    def get_logging_engine(this):
        return get_neko_logging_agent(this.platform)



    def get_tester(this, benchdict, anchor_dict, prefix="", k=1):
        ta = neko_test_mission_agent;
        return {
            "agent": ta,
            "params": {
                ta.PARAM_test_benches: benchdict,
                ta.AGENT_reporter_dict: this.get_reporters(prefix),
                ta.PARAM_test_variant: "",
                ta.AGENT_tester_agent: this.get_testing_fp_agent(prefix, anchor_dict, k=k),
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
    def get_testers(this, dict_name_bench, anchor_dict, prefix=""):
        rd = {};
        for n in dict_name_bench:
            rd[n] = this.get_tester(dict_name_bench[n], anchor_dict, prefix, k=1);
        return rd;
    # don't engage moe if not trained with it!!! dummy!!!
    def get_imbased_tester(this, anchor_dict,logpath,saveprfx, prefix="", k=1):
        tfp=this.get_testing_fp_agent(prefix, anchor_dict, k=k);
        return this.dbge.as_athena(tfp,{},anchor_dict,logpath, saveprfx, prefix);

    # don't engage moe if not trained with it!!! dummy!!!


    def get_debugger(this, benchdict, anchor_dict, logpath, saveprfx, prefix="", core_dump=False, forcek=1):
        cta = this.get_testing_fp_agent(prefix, anchor_dict, k=forcek);
        return this.dbge.as_debugger(cta,benchdict,anchor_dict,logpath,saveprfx,prefix, core_dump, forcek);

    def get_debuggers(this, dict_name_bench, anchor_dict, logpath, saveprfx, prefix="", core_dump=False, forcek=1):
        rd = {}
        for n in dict_name_bench:
            rd[n] = this.get_debugger(dict_name_bench[n], anchor_dict, logpath, saveprfx, prefix, core_dump, forcek);

        return rd;

    def get_trainer_param(this,val_dict: Dict, anchor, qname, ecnt=5, icnt=200000):
        e = neko_trainer_agent;
        return{
            "routine_names": [this.VAN.ANAME_core_routine],
            # expand this list if you want to pull smarter cotraining plans.
            "routine_dict": {this.VAN.ANAME_core_routine: this.get_training_agent(anchor, qname, prefixes=None)},
            # well we are not doing sdmi here :-) maybe relavant if in the future we need to train with data on HDDs, but...
            "pretest_names": [],
            "pretest_dict": {
            },
            "tester_names": list(val_dict.keys()),
            "tester_dict": val_dict,
            "posttest_names": [],
            "posttest_dict": {

            },
            "epoch_logger_names": [],
            "epoch_logger_dict": {},
            e.PARAM_iter_logger_names: [this.VAN.ANAME_training_logger],
            e.PARAM_iter_logger_dict: {this.VAN.ANAME_training_logger: this.get_logging_engine()},
            "epoch_cnt": ecnt,
            "iter_cnt": icnt,
            "devices": this.platform.devices
        }

    # allows partial training by tags. Used for lm rn. Later maybe batchnorm, incremental head addition, etc.
    def get_trainer(this, val_dict: Dict, anchor, qname, ecnt=5, icnt=200000,tags=None):
        # set up trainer.
        e = neko_trainer_agent;
        trainer_param=this.get_trainer_param(val_dict,anchor,qname,ecnt,icnt);
        if(tags is None):
            trainer_param[neko_trainer_agent.PARAM_trainable_tags]="NEP_skipped_NEP";
        else:
            trainer_param[neko_trainer_agent.PARAM_trainable_tags]=tags;
        # print(json.dumps(trainer_param,default=lambda x:  str(x)));
        trainer = e(
            trainer_param
        )
        return trainer;

