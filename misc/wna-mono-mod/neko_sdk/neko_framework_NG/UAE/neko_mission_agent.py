import glob
import os.path

import cv2
import numpy as np
import torch
from easydict import EasyDict

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from neko_sdk.neko_framework_NG.modules.neko_label_sampler_NG import neko_abstract_sideinfo_source
from neko_sdk.neko_framework_NG.names import default_variable_names as dvn
from neko_sdk.neko_framework_NG.names import default_param_names as dpn
from osocrNG.athena.common.analyze_folder import bootstrap_folder_NG

class neko_pretest_agent(neko_abstract_sync_agent):
    def take_action(this, _: neko_workspace, environment: neko_environment):
        environment.modset.eval_mode();


class neko_posttest_agent(neko_abstract_sync_agent):
    def take_action(this, _: neko_workspace, environment: neko_environment):
        environment.modset.train_mode();

from osocrNG.data_utils.data_agents.testing_dataset_agents import testing_cached_proto_loading_agent,testing_cached_proto_loading_agent_multiple
class neko_abstract_mission_agent(neko_abstract_sync_agent):
    MOD_proto_mvn_name="proto_mvn_name";
    MOD_prototyper_name="prototyper_name";
    AGENT_tester_agent="tester";
    AGENT_reporter_dict="reporters";
    OUTPUT_raw_id_name=dvn.raw_id_name;
    OUTPUT_raw_image_name=dvn.raw_image_name;
    OUTPUT_raw_label_name = dvn.raw_label_name;
    PARAM_test_benches="tests";
    PARAM_test_variant="variants";
    # these are level 2 parameters, which means you will need if you are forging deeper stuff.
    # Generally you ignore them
    PARAM_2_test_bench_dataset="data";
    PARAM_2_test_bench_meta = "meta";
    def meta_of_test(this,test):
        return this.tests[this.PARAM_test_benches][test][this.PARAM_2_test_bench_meta];
    def data_of_test(this,test):
        return this.tests[this.PARAM_test_benches][test][this.PARAM_2_test_bench_dataset];
    def hasunk(this,test):
        return this.tests[this.PARAM_2_test_bench_meta][this.meta_of_test(test)]["has_unk"];
    def set_proto_io(this,iocvt_dict,modcvt_dict):
        pass;
    def set_proto_loader(this,param):
        pass;
    def arm_cached_protos(this,workspace,protodict,test):
        return workspace,environment;
    def cache_protos(this, environment):
        with torch.no_grad():
            protodict = this.proto_loader.cache_prototypes(environment);
        return protodict;
    def setup(this, param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.internal_dict = EasyDict();

        this.omods = EasyDict();
        this.mnames = EasyDict();
        this.set_mod_io(param[dpn.iocvt_dict], param[dpn.modcvt_dict]);
        this.set_proto_loader(param);
        this.tests = param[this.PARAM_test_benches];
        this.variant=neko_get_arg(this.PARAM_test_variant,param,"");
        this.test_data_holders = {};

        if(len(this.tests)):
            dataparam = neko_get_arg(this.PARAM_2_test_bench_dataset, this.tests);
            for t in dataparam:
                this.test_data_holders[t] = dataparam[t];
        else:
            this.test_data_holders={};
        ecfg = neko_get_arg(this.AGENT_tester_agent, param);
        this.tester = ecfg["agent"](ecfg["params"]);
        rcfg = neko_get_arg(this.AGENT_reporter_dict, param);
        this.reporters = {};
        for r in rcfg:
            this.reporters[r] = rcfg[r]["agent"](rcfg[r]["params"]);
        pass;
    def dump_proto(this, _: neko_workspace, environment: neko_environment,path):
        protodict = this.cache_protos(environment);
        torch.save(protodict,path);

    def take_action_multibatch(this, _: neko_workspace, environment: neko_environment, bs=160):

        protodict = this.cache_protos(environment);
        trd = {};

        for test in this.tests[this.PARAM_test_benches]:
            trd[test] = {}
            for r in this.reporters:
                this.reporters[r].reset(test + this.variant, this.hasunk(test));
            d = this.test_data_holders[this.data_of_test(test)];
            for i in range(0, len(d), bs):
                aret = [];
                aids = [];
                for j in range(i, min(len(d), i + bs)):
                    id = {"id": j + 1};
                    ret = d.fetch_item(id);
                    if (ret is not None):
                        aret.append(ret);
                        aids.append(id);
                workspace = neko_workspace();
                workspace.inter_dict[this.image_name] = [np.array(r[RN.IMAGE]) for r in aret];
                workspace.inter_dict[this.id_name] = [id for id in aids];
                if (RN.LABEL in aret[0]):
                    workspace.inter_dict[this.text_label_name] = [r[RN.LABEL] for r in aret];
                workspace = this.arm_cached_protos(workspace, protodict, test);

                this.tester.take_action(workspace, environment);
                for r in this.reporters:
                    this.reporters[r].take_action(workspace, environment);
            for r in this.reporters:
                trd[test][r] = this.reporters[r].report(environment);
        return trd;

    def take_action(this, _: neko_workspace, environment: neko_environment):
        protodict=this.cache_protos(environment);
        for test in this.tests[this.PARAM_test_benches]:
            for r in this.reporters:
                this.reporters[r].reset(test,this.tests[this.PARAM_2_test_bench_meta][this.meta_of_test(test)]["has_unk"]);
            d = this.test_data_holders[this.tests[this.PARAM_test_benches][test][this.PARAM_2_test_bench_dataset]];
            for i in range(len(d)):
                id={"id": i + 1};
                ret = d.fetch_item(id);
                workspace = neko_workspace();
                if (ret is None):
                    continue;
                workspace.inter_dict[this.image_name] = [np.array(ret[RN.IMAGE])];
                workspace.inter_dict[this.id_name]=[id];
                if(RN.LABEL in ret):
                    workspace.inter_dict[this.text_label_name] = [ret[RN.LABEL]];
                workspace=this.arm_cached_protos(workspace,protodict,test);

                this.tester.take_action(workspace, environment);
                for r in this.reporters:
                    this.reporters[r].take_action(workspace, environment);
            rd={};
            for r in this.reporters:
                this.reporters[r].report(environment);

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto_mvn_name=this.register(this.MOD_proto_mvn_name,modcvt_dict,this.mnames);
        this.prototyper_name=this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.id_name=this.register(this.OUTPUT_raw_id_name,iocvt_dict,this.output_dict);
        this.image_name=this.register(this.OUTPUT_raw_image_name,iocvt_dict,this.output_dict);
        this.text_label_name=this.register(this.OUTPUT_raw_label_name,iocvt_dict,this.output_dict);
        this.set_proto_io(iocvt_dict,modcvt_dict);


class neko_test_mission_agent(neko_abstract_mission_agent):
    OUTPUT_tensor_proto_vec_name = dvn.tensor_proto_vec_name;
    OUTPUT_tensor_proto_img_name = dvn.tensor_proto_img_name;
    OUTPUT_weightroto_label_name = dvn.proto_label_name;
    OUTPUT_meta_dict="meta_dict";
    OUTPUT_split_flare="split_flair"

    OUTPUT_tdict_name = dvn.tdict_name;



    def set_proto_io(this, iocvt_dict, modcvt_dict):
        this.meta_dict=this.register(this.OUTPUT_meta_dict,iocvt_dict,this.output_dict,"meta_dict");
        this.proto_name = this.register(this.OUTPUT_tensor_proto_vec_name, iocvt_dict, this.output_dict);
        this.proto_img_name=this.register(this.OUTPUT_tensor_proto_img_name,iocvt_dict,this.output_dict);
        this.proto_label_name = this.register(this.OUTPUT_weightroto_label_name, iocvt_dict, this.output_dict);
        this.tdict_name = this.register(this.OUTPUT_tdict_name, iocvt_dict, this.output_dict);
        this.split_flair = this.register(this.OUTPUT_split_flare, iocvt_dict, this.output_dict,"benchmark_name");


    def arm_cached_protos(this, workspace, protodict, test):
        testmeta=protodict[this.meta_of_test(test)];
        workspace.inter_dict[this.proto_name] = testmeta[
            this.proto_name];
        workspace.inter_dict[this.proto_img_name]=testmeta[this.proto_img_name]
        workspace.inter_dict[this.proto_label_name] = testmeta[
            this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = testmeta[
            this.tdict_name];
        workspace.inter_dict[this.split_flair]=test;
        # hack the meta dict in.
        # we will revise this in 32x
        workspace.inter_dict[this.meta_dict]={"main":this.tests[this.PARAM_2_test_bench_meta][this.meta_of_test(test)]["meta_path"]};

        return workspace;

    def set_proto_loader(this, param):
        this.proto_loader = testing_cached_proto_loading_agent(
            {
                "meta": param["tests"]["meta"],
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_weightroto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_img_name: this.proto_img_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )

class neko_test_mission_agent_single_im(neko_test_mission_agent):

    def arm_path(this,spath,dpath,lang,pfx="png"):
        files, v2ptfolder, sfolder, dfolder=\
            bootstrap_folder_NG(spath,dpath,lang,"*."+pfx);
        this.testname=lang;
        this.files=files;
        this.tests={

            this.PARAM_test_benches:{
                this.testname:{
                    this.PARAM_2_test_bench_meta:lang
                }
            }

        };
        this.set_proto_loader({
            this.PARAM_test_benches:{
                this.PARAM_2_test_bench_meta: {
                    this.testname:{
                        neko_abstract_sideinfo_source.PARAM_meta_path:v2ptfolder
                    }},
                this.PARAM_2_test_bench_dataset:{},
                this.PARAM_test_benches:{},
                }});
    def set_proto_loader(this, param):
        if(len(param[this.PARAM_test_benches])==0):
            return ; # do not load
        this.proto_loader = testing_cached_proto_loading_agent(
            {
                "meta": param[this.PARAM_test_benches]["meta"],
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_weightroto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_img_name: this.proto_img_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )
    def take_action(this, _: neko_workspace, environment: neko_environment):
        protodict=this.cache_protos(environment);
        for r in this.reporters:
            this.reporters[r].reset(this.testname,False);
        for i in range(len(this.files)):
            ret ={
                RN.IMAGE:cv2.imread(this.files[i]),
                RN.UID:{"id":os.path.basename(this.files[i])},
            }
            workspace = neko_workspace();
            if (ret is None):
                continue;
            workspace.inter_dict[this.image_name] = [np.array(ret[RN.IMAGE])];
            workspace.inter_dict[this.id_name]=[ret[ RN.UID]];
            if(RN.LABEL in ret):
                workspace.inter_dict[this.text_label_name] = [ret[RN.LABEL]];
            workspace=this.arm_cached_protos(workspace,protodict,this.testname);
            this.tester.take_action(workspace, environment);
            for r in this.reporters:
                this.reporters[r].take_action(workspace, environment);
        for r in this.reporters:
            this.reporters[r].report(environment);

    def arm_cached_protos(this, workspace, protodict, test):
        testmeta=protodict[this.meta_of_test(test)];
        workspace.inter_dict[this.proto_name] = testmeta[
            this.proto_name];
        workspace.inter_dict[this.proto_img_name]=testmeta[this.proto_img_name]
        workspace.inter_dict[this.proto_label_name] = testmeta[
            this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = testmeta[
            this.tdict_name];
        workspace.inter_dict[this.split_flair]=test;
        return workspace;



class neko_test_mission_agent_vbd(neko_abstract_mission_agent):
    OUTPUT_tensor_proto_vec_name = dvn.tensor_proto_vec_name;
    OUTPUT_weightroto_label_name = dvn.proto_label_name;
    OUTPUT_tdict_name = dvn.tdict_name;
    OUTPUT_rotated_proto_tensor_name = dvn.rotated_tensor_proto_vec_name;
    OUTPUT_split_flare="split_flair"


    def set_proto_io(this, iocvt_dict, modcvt_dict):
        this.rotated_proto_name = this.register(this.OUTPUT_rotated_proto_tensor_name, iocvt_dict, this.output_dict);
        this.proto_name=this.register(this.OUTPUT_tensor_proto_vec_name,iocvt_dict,this.output_dict);
        this.proto_label_name = this.register(this.OUTPUT_weightroto_label_name, iocvt_dict, this.output_dict);
        this.tdict_name = this.register(this.OUTPUT_tdict_name, iocvt_dict, this.output_dict);
        this.split_flair = this.register(this.OUTPUT_split_flare, iocvt_dict, this.output_dict);

    def arm_cached_protos(this, workspace, protodict, test):
        testmeta=protodict[this.meta_of_test(test)];
        workspace.inter_dict[this.proto_name] = testmeta[
            this.proto_name];
        workspace.inter_dict[this.rotated_proto_name] =testmeta[
            this.rotated_proto_name];
        workspace.inter_dict[this.proto_label_name] = testmeta[
            this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = testmeta[
            this.tdict_name];
        workspace.inter_dict[this.split_flair]=test;
        return workspace,environment;

    def set_proto_loader(this, param):
        this.proto_loader = testing_cached_proto_loading_agent(
            {
                testing_cached_proto_loading_agent.PARAM_meta: param["tests"]["meta"],
                testing_cached_proto_loading_agent.PARAM_possible_rotation:[0,1,2,3],
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_rotated_tensor_proto_vec_name:this.rotated_proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_weightroto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )

#
#
# class neko_test_mission_agent_multi_sel_proto(neko_abstract_sync_agent):
#     def set_mod_io(this, iocvt_dict, modcvt_dict):
#         this.mnames.proto_mvn_name=modcvt_dict["proto_mvn_name"];
#         this.mnames.prototyper_name=modcvt_dict["prototyper_name"];
#         this.output_dict.image_name=iocvt_dict[dvn.raw_image_name];
#         this.output_dict.text_label_name=iocvt_dict[dvn.raw_label_name];
#         this.output_dict.proto_name=iocvt_dict[dvn.tensor_proto_vec_name];
#         this.output_dict.proto_label_name=iocvt_dict[dvn.proto_label_name];
#         this.output_dict.tdict_name=iocvt_dict[dvn.tdict_name];
#
#
#     def setup(this, param):
#         this.input_dict = EasyDict();
#         this.output_dict = EasyDict();
#         this.internal_dict=EasyDict();
#
#         this.omods = EasyDict();
#         this.mnames = EasyDict();
#         this.set_mod_io(param["iocvt_dict"], param["modcvt_dict"]);
#         test_data_holders:dict[neko_lmdb_holder];
#         this.proto_loader=testing_cached_proto_loading_agent_multiple(
#             {
#                 "meta":param["tests"]["meta"],
#                 "iocvt_dict":{
#                     "meta_key_name": "NEP_skipped_NEP",
#                     "tensor_proto_vec_name": this.output_dict.proto_name,
#                     "plabel_name": this.output_dict.proto_label_name,
#                     "tdict_name": this.output_dict.tdict_name,
#                 },
#                 "modcvt_dict":{
#                     "proto_mvn_name": this.mnames.proto_mvn_name,
#                     "prototyper_name":this.mnames.prototyper_names
#                 },
#             }
#         );
#         this.tests=param["tests"]
#         this.test_data_holders={};
#         dataparam=neko_get_arg("data", this.tests);
#         for t in dataparam:
#             this.test_data_holders[t]=dataparam[t];
#         ecfg=neko_get_arg("tester",param);
#         this.tester=ecfg["agent"](ecfg["params"]);
#         rcfg=neko_get_arg("reporters",param);
#         this.reporters={};
#         for r in rcfg:
#             this.reporters[r]=rcfg[r]["agent"](rcfg[r]["params"]);
#         pass;
#
#     def take_action(this, _: neko_workspace, environment: neko_environment):
#         protodict=this.proto_loader.cache_prototypes(environment);
#         for test in this.tests["tests"]:
#             for r in this.reporters:
#                 this.reporters[r].reset(test);
#             d=this.test_data_holders[ this.tests["tests"][test]["data"]];
#             for i in range(len(d)):
#                 ret=d.fetch_item({"id":i+1});
#                 workspace=neko_workspace();
#                 workspace.inter_dict[this.output_dict.image_name] = [np.array(ret["image"])];
#                 workspace.inter_dict[this.output_dict.text_label_name] = [ret["label"]];
#                 workspace.inter_dict[this.output_dict.proto_name]=protodict[this.meta_of_benchmark(test)]["prototypes"];
#                 workspace.inter_dict[this.output_dict.proto_label_name]=protodict[this.meta_of_benchmark(test)]["plabels"];
#                 workspace.inter_dict[this.output_dict.tdict_name] = protodict[this.meta_of_benchmark(test)]["tdict"];
#                 this.tester.take_action(workspace,environment);
#                 for r in this.reporters:
#                     this.reporters[r].take_action(workspace,environment);
#             for r in this.reporters:
#                 this.reporters[r].report(environment);
#
