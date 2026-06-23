from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from osocrNG.data_utils.data_agents.testing_dataset_agents import testing_cached_proto_loading_agent
from osocrNG.names import default_ocr_variable_names as dvn
from easydict import EasyDict
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import numpy as np
from osocrNG.names import default_ocr_param_names as dpn
import torch

class neko_sync_prediction_agent(neko_abstract_sync_agent):
    MOD_proto_mvn_name="proto_mvn_name";
    MOD_prototyper_name="prototyper_name";
    OUTPUT_raw_image_name=dvn.raw_image_name;
    OUTPUT_raw_label_name = dvn.raw_label_name;

    def setup(this, param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.internal_dict = EasyDict();

        this.omods = EasyDict();
        this.mnames = EasyDict();
        this.set_mod_io(param[dpn.iocvt_dict], param[dpn.modcvt_dict]);
        this.set_proto_loader(param);

        ecfg = neko_get_arg("tester", param);
        this.tester = ecfg["agent"](ecfg["params"]);

        pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto_mvn_name=this.register(this.MOD_proto_mvn_name,modcvt_dict,this.mnames);
        this.prototyper_name=this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.image_name=this.register(this.OUTPUT_raw_image_name,iocvt_dict,this.output_dict);
        this.text_label_name=this.register(this.OUTPUT_raw_label_name,iocvt_dict,this.output_dict);
        this.set_proto_io(iocvt_dict,modcvt_dict);


    def arm_cached_protos(this, workspace):
        workspace.inter_dict[this.proto_name] = this.protodict[this.proto_name];
        workspace.inter_dict[this.proto_label_name] = this.protodict[this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = this.protodict[this.tdict_name];
        return workspace,environment;

    def cache_proto(this, meta_path,environment,case_sensitive=False,use_sp=False):
        proto_loader = testing_cached_proto_loading_agent(
            {
                "meta": meta_path,
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_weightroto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )
        with torch.no_grad():
            protodict = proto_loader.cache_prototypes(environment);
        return protodict;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        workspace = this.arm_cached_protos(workspace);
        this.tester.take_action(workspace, environment);

    def test_image(this,image:np.array,environment:neko_environment):
        workspace = neko_workspace();
        workspace.inter_dict[this.image_name] = [image];
        this.take_action(workspace,environment);


    def setup(this, param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.internal_dict = EasyDict();
        this.omods = EasyDict();
        this.mnames = EasyDict();

        this.set_mod_io(param[dpn.iocvt_dict], param[dpn.modcvt_dict]);
        this.tests = param["tests"]
        this.test_data_holders = {};
        dataparam = neko_get_arg("data", this.tests);
        for t in dataparam:
            this.test_data_holders[t] = dataparam[t];
        ecfg = neko_get_arg("tester", param);
        this.tester = ecfg["agent"](ecfg["params"]);

        pass;
