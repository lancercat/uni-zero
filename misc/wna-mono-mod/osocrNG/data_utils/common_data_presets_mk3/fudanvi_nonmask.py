import os.path
import time

import cv2
import tqdm

from osocr_tasks.ds_paths import get_fudanvi_document_test,get_fudanvi_web_test,get_fudanvi_scene_test,get_fudanvi_hwdb_test

from osocrNG.data_utils.data_agents.multilmdb_anchor_balanced_mixed_agent import neko_balance_fetching_and_mixing_agent
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocrNG.data_utils.holder_cfg import (
    get_fudanvi_hwdb_tr_holder,get_fudanvi_hwdb_te_holder, get_fudanvi_hwdb_tr_metav2, get_fudanvi_hwdb_test_metav2,
    get_fudanvi_doc_tr_holder,get_fudanvi_doc_te_holder, get_fudanvi_doc_tr_metav2, get_fudanvi_doc_test_metav2,
    get_fudanvi_scene_tr_holder,get_fudanvi_scene_te_holder, get_fudanvi_scene_tr_metav2, get_fudanvi_scene_test_metav2,
    get_fudanvi_web_tr_holder, get_fudanvi_web_te_holder, get_fudanvi_web_tr_metav2, get_fudanvi_web_test_metav2,
    get_fudanvi_all_tr_metav2,get_fudanvi_all_test_metav2,get_fudan_github_test_metav2)
from osocrNG.data_utils.common_data_presets_mk3.factory import abstract_mk3_data_factory

# using factory class to avoid wild strings.
# you can still use wild strings to call the factory,
# but now you can not to.
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent,augment_agent_abinet,augment_agent_abinet_fixed


class fudavi_mk3_data_factory(abstract_mk3_data_factory):
    DSD={
        "scene_tr":get_fudanvi_scene_tr_holder,
        "doc_tr":get_fudanvi_doc_tr_holder,
        "hw_tr":get_fudanvi_hwdb_tr_holder,
        "web_tr":get_fudanvi_web_tr_holder,
    }
    TEDSD={
        "scene_te": get_fudanvi_scene_te_holder,
        "doc_te": get_fudanvi_doc_te_holder,
        "hw_te": get_fudanvi_hwdb_te_holder,
        "web_te": get_fudanvi_web_te_holder,
    }
    TEMETA={
        "scene":get_fudanvi_scene_test_metav2,
        "web": get_fudanvi_web_test_metav2,
        "doc": get_fudanvi_doc_test_metav2,
        "hwdb": get_fudanvi_hwdb_test_metav2,
        "all":get_fudanvi_all_test_metav2,
        "fudan": get_fudan_github_test_metav2,
    }
    CFG={
        "scene":{
            "trmeta":get_fudanvi_scene_tr_metav2,
            "trds": ["scene_tr"],
            # "teds" :["scene_te"],
            # "temeta":get_fudanvi_scene_test_metav2
        },
        "nohwdb":
        {
            "trmeta": get_fudanvi_all_tr_metav2,
            "trds": ["scene_tr", "doc_tr", "web_tr"],
        },
        "web": {
            "trmeta":get_fudanvi_web_tr_metav2,
            "trds": ["web_tr"],
        },
        "all":{
            "trmeta": get_fudanvi_all_tr_metav2,
            "trds": ["scene_tr","doc_tr","hw_tr","web_tr"],
            # "teds": ["scene_te","doc_te","hw_te","web_te"],
            # "temeta": get_fudanvi_all_test_metav2
        },
        "hw_tr":{
            "trmeta": get_fudanvi_all_tr_metav2,
            "trds": ["hw_tr"],
            # "teds": ["scene_te","doc_te","hw_te","web_te"],
            # "temeta": get_fudanvi_all_test_metav2
        }
    };
    TET={
        "scene" :{
                "data": "scene_te",
                "meta": "scene"
        },
        "scene_fudan":{
            "data": "scene_te",
            "meta": "fudan"
        },
        "web": {
            "data": "web_te",
            "meta": "web"
        },
        "web_fudan": {
            "data": "web_te",
            "meta": "fudan"
        },
        "doc": {
            "data": "doc_te",
            "meta": "doc"
        },
        "doc_fudan": {
            "data": "doc_te",
            "meta": "fudan"
        },
        "hwdb": {
            "data": "hw_te",
            "meta": "hwdb"
        },
        "hwdb_fudan": {
            "data": "hw_te",
            "meta": "fudan"
        },
    }
    @classmethod
    def get_all_test_data_metav2(cls,dataroot):
        rd={
            "meta":{},
            "data":{},
        };
        for i in cls.TEMETA:
            rd["meta"][i]= {
                "meta_path":cls.TEMETA[i](dataroot),
                "case_sensitive": False,
                "has_unk": False
                   };
        for d in cls.TEDSD:
            rd["data"][d]=cls.TEDSD[d](dataroot);
        return rd;

    @classmethod
    def get_fudan_test(cls,dataroot,keys):
        rd=cls.get_all_test_data_metav2(dataroot)
        rd["tests"]={
        };
        for k in keys:
            rd["tests"][k]=cls.TET[k];
        return rd

    @classmethod
    def get_fudan_v2_training(cls,dataroot,anchor_dict,data_queue_name,key):
        he=neko_named_multi_source_holder;
        dd={
            he.PARAM_sources:[],
            he.PARAM_sourced:{}
            };
        for di in cls.CFG[key]["trds"]:
            dd[he.PARAM_sources].append(di);
            dd[he.PARAM_sourced][di]=cls.DSD[di](dataroot);
        holder=neko_named_multi_source_holder(dd);
        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"fudan-"+key);


    @classmethod
    def arm_fudan_hydra_v2(cls,agent_dict,qdict,params,trkey):
        pre_aug_qname = params[cls.PARAM_preaug_data_queue_name];
        da = cls.get_fudan_v2_training(params[cls.PARAM_dataroot],params[cls.PARAM_anchor_dict],pre_aug_qname,trkey);
        meta=cls.CFG[trkey]["trmeta"](params[cls.PARAM_dataroot]);
        agent_dict,qdict=cls.arm_training_data(da,agent_dict,qdict,params);
        return agent_dict,qdict,meta;




class fudavi_mk3_data_factory_scene(fudavi_mk3_data_factory):
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trkey = "scene";
        tekeys = ["scene","scene_fudan"];
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_fudan_hydra_v2(trad, trqd,
                                                 {
                                                     cls.PARAM_dataroot: data_root,
                                                     cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                     cls.EXPORT_data_queue_name: queue_name,
                                                     cls.PARAM_anchor_dict: anchor_dict,
                                                 }, trkey
                                                 );
        tedd = cls.get_fudan_test(data_root, tekeys);
        return trad, trqd, trm, tedd;
class fudavi_mk3_data_factory_sceneAA(fudavi_mk3_data_factory_scene):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet;
    @classmethod
    def get_mk3_benchmark_plus(cls, data_root, anchor_dict, queue_name):
        return cls.get_mk3_benchmark(data_root, anchor_dict, queue_name);

class fudavi_mk3_data_factory_sceneAAF(fudavi_mk3_data_factory_scene):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_fixed;
    @classmethod
    def get_mk3_benchmark_plus(cls, data_root, anchor_dict, queue_name):
        return cls.get_mk3_benchmark(data_root, anchor_dict, queue_name);


class fudavi_mk3_data_factory_all(fudavi_mk3_data_factory):
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trkey = "all";
        tekeys = ["scene_fudan","web_fudan","doc_fudan","hwdb_fudan"];
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_fudan_hydra_v2(trad, trqd,
                                                 {
                                                     cls.PARAM_dataroot: data_root,
                                                     cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                     cls.EXPORT_data_queue_name: queue_name,
                                                     cls.PARAM_anchor_dict: anchor_dict,
                                                 }, trkey
                                                 );
        tedd = cls.get_fudan_test(data_root, tekeys);
        return trad, trqd, trm, tedd;



class fudavi_mk3_data_factory_nohwdb(fudavi_mk3_data_factory):
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trkey = "nohwdb";
        tekeys = ["scene_fudan","web_fudan","doc_fudan"];
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_fudan_hydra_v2(trad, trqd,
                                                 {
                                                     cls.PARAM_dataroot: data_root,
                                                     cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                     cls.EXPORT_data_queue_name: queue_name,
                                                     cls.PARAM_anchor_dict: anchor_dict,
                                                 }, trkey
                                                 );
        tedd = cls.get_fudan_test(data_root, tekeys);
        return trad, trqd, trm, tedd;
class fudavi_mk3_data_factory_doc(fudavi_mk3_data_factory):
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trkey = "scene";
        tekeys = ["scene_fudan"];
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_fudan_hydra_v2(trad, trqd,
                                                 {
                                                     cls.PARAM_dataroot: data_root,
                                                     cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                     cls.EXPORT_data_queue_name: queue_name,
                                                     cls.PARAM_anchor_dict: anchor_dict,
                                                 }, trkey
                                                 );
        tedd = cls.get_fudan_test(data_root, tekeys);
        return trad, trqd, trm, tedd;


class fudavi_mk3_data_factory_hwd(fudavi_mk3_data_factory):
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trkey = "hw_tr";
        tekeys = ["hwdb_fudan"];
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_fudan_hydra_v2(trad, trqd,
                                                 {
                                                     cls.PARAM_dataroot: data_root,
                                                     cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                     cls.EXPORT_data_queue_name: queue_name,
                                                     cls.PARAM_anchor_dict: anchor_dict,
                                                 }, trkey
                                                 );
        tedd = cls.get_fudan_test(data_root, tekeys);
        return trad, trqd, trm, tedd;