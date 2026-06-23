import os.path
import time

import cv2
import tqdm

from osocr_tasks.ds_paths import get_iiit5k, get_SVT, get_cute, \
    get_IC03_867, get_IC13_1015, get_IC15_2077, get_SVTP
from osocrNG.data_utils.data_agents.multilmdb_anchor_balanced_mixed_agent import neko_balance_fetching_and_mixing_agent
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocrNG.data_utils.holder_cfg import get_cvpr_2016_holder,get_nips_2014_holder, get_abi_mj_te_holder,get_abi_mj_tr_holder,get_abi_st_holder,get_abi_mj_val_holder
from osocrNG.data_utils.common_data_presets_mk3.factory import abstract_mk3_data_factory
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent,augment_agent_abinet,augment_agent_abinet_fixed

# using factory class to avoid wild strings.
# you can still use wild strings to call the factory,
# but now you can not to.
class mjst_mk3_data_factory(abstract_mk3_data_factory):
    @classmethod
    def get_eng_test_v1(cls,dataroot,v2h=-9):
        FNS=[get_iiit5k,get_SVT,get_cute,get_IC03_867,get_IC13_1015,get_IC15_2077,get_SVTP];
        NMS=["IIIT5k"];
        metadict={
            "EN":
                  {"meta_path": os.path.join(dataroot, "dictsv2", "dab62cased"),
                   "case_sensitive": False,
                   "has_unk": False
                   }
                  };
        datadict={
        };
        for d,n in zip(FNS,NMS):
            datadict[n]=neko_lmdb_holder({"root": d(dataroot),cls.PARAM_v2h:v2h})

        test_dict={};
        for d in datadict:
            for m in metadict:
                test_dict[d+"-close"]={
                    "data":d,
                    "meta":m,
                }
        return {
            "meta":metadict,
            "data":datadict,
            "tests":test_dict
        };

    @classmethod
    def get_mjst_v1_balance_mix(cls,dataroot,anchor_dict,data_queue_name):
        he=neko_named_multi_source_holder;
        holder=neko_named_multi_source_holder(
            {
                he.PARAM_sources:["CVPR2016","NIPS2014"],
                he.PARAM_sourced:{
                    "CVPR2016": get_cvpr_2016_holder(dataroot),
                    "NIPS2014": get_nips_2014_holder(dataroot),
                }
            }
        );

        return cls.get_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"mjst-")
    @classmethod
    def get_mjst_v2_balance_mix(cls,dataroot,anchor_dict,data_queue_name):
        he=neko_named_multi_source_holder;
        holder=neko_named_multi_source_holder(
            {
                he.PARAM_sources:["CVPR2016","NIPS2014"],
                he.PARAM_sourced:{
                    "CVPR2016": get_cvpr_2016_holder(dataroot),
                    "NIPS2014": get_nips_2014_holder(dataroot),
                }
            }
        );

        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"mjst-")



    @classmethod
    def arm_mjst_hydra_v1(cls,agent_dict,qdict,params):
        pre_aug_qname = params[cls.PARAM_preaug_data_queue_name];
        da = cls.get_mjst_v1_balance_mix(params[cls.PARAM_dataroot],params[cls.PARAM_anchor_dict],pre_aug_qname);
        meta=os.path.join(params[cls.PARAM_dataroot], "dictsv2", "dab62cased");
        agent_dict,qdict=cls.arm_training_data(da,agent_dict,qdict,params);
        return agent_dict,qdict,meta;
    @classmethod
    def arm_mjst_hydra_v2(cls,agent_dict,qdict,params):
        pre_aug_qname = params[cls.PARAM_preaug_data_queue_name];
        da = cls.get_mjst_v2_balance_mix(params[cls.PARAM_dataroot],params[cls.PARAM_anchor_dict],pre_aug_qname);
        meta=os.path.join(params[cls.PARAM_dataroot], "dictsv2", "dab62cased");
        agent_dict,qdict=cls.arm_training_data(da,agent_dict,qdict,params);
        return agent_dict,qdict,meta;
    @classmethod
    def get_benchmark(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_mjst_hydra_v1(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_eng_test_v1(data_root);
        return trad,trqd,trm,tedd;



    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_mjst_hydra_v2(trad, trqd,
                                                {
                                                    cls.PARAM_dataroot: data_root,
                                                    cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                    cls.EXPORT_data_queue_name: queue_name,
                                                    cls.PARAM_anchor_dict: anchor_dict,
                                                }
                                                );
        tedd = cls.get_eng_test_v1(data_root);
        return trad, trqd, trm, tedd;
class mjst_mk3_data_factory_abi(mjst_mk3_data_factory):
    @classmethod
    def get_mjst_v2_balance_mixabi(cls,dataroot,anchor_dict,data_queue_name):
        he=neko_named_multi_source_holder;
        holder=neko_named_multi_source_holder(
            {
                he.PARAM_sources:["MJ","MJ-val","MJ-te","ST"],
                he.PARAM_sourced:{
                    "MJ": get_abi_mj_tr_holder(dataroot),
                    "MJ-val": get_abi_mj_tr_holder(dataroot),
                    "MJ-te": get_abi_mj_te_holder(dataroot),
                    "ST": get_abi_st_holder(dataroot),
                }
            }
        );

        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"mjstABI-")

    @classmethod
    def arm_mjst_hydra_v2abi(cls, agent_dict, qdict, params):
        pre_aug_qname = params[cls.PARAM_preaug_data_queue_name];
        da = cls.get_mjst_v2_balance_mixabi(params[cls.PARAM_dataroot], params[cls.PARAM_anchor_dict], pre_aug_qname);
        meta = os.path.join(params[cls.PARAM_dataroot], "dictsv2", "dab62cased");
        agent_dict, qdict = cls.arm_training_data(da, agent_dict, qdict, params);
        return agent_dict, qdict, meta;
    @classmethod
    def get_mk3_benchmark(cls, data_root, anchor_dict, queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_mjst_hydra_v2abi(trad, trqd,
                                                   {
                                                       cls.PARAM_dataroot: data_root,
                                                       cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                       cls.EXPORT_data_queue_name: queue_name,
                                                       cls.PARAM_anchor_dict: anchor_dict,
                                                   }
                                                   );
        tedd = cls.get_eng_test_v1(data_root);
        return trad, trqd, trm, tedd;

class mjst_mk3_data_factoryAA(mjst_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet;

class mjst_mk3_data_factory_abiAA(mjst_mk3_data_factory_abi):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet;
    @classmethod
    def get_mk3_benchmark_plus(cls, data_root, anchor_dict, queue_name):
        return cls.get_mk3_benchmark( data_root, anchor_dict, queue_name);

class mjst_mk3_data_factory_abiAAF(mjst_mk3_data_factory_abi):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_fixed;
    @classmethod
    def get_mk3_benchmark_plus(cls, data_root, anchor_dict, queue_name):
        return cls.get_mk3_benchmark( data_root, anchor_dict, queue_name);



if __name__ == '__main__':
    from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
    from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent

    ad,qd={},{};
    from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1v_6_05 as two_hori_main
    anchors=two_hori_main();

    de=mjst_mk3_data_factory;
    ad,qd,meta=mjst_mk3_data_factory.arm_mjst_hydra_v1(ad,qd,
                                               {
                                                   de.PARAM_dataroot:"/home/lasercat/ssddata",
                                                   de.PARAM_preaug_data_queue_name:"preaug_q",
                                                   de.EXPORT_data_queue_name: "dq",
                                                   de.PARAM_anchor_dict:anchors,
                                               }
                                               );
    from multiprocessing import Queue as mpQueue
    for qk in qd:
        qd[qk]=mpQueue(maxsize=9);
    e=neko_environment(queue_dict=qd);
    for ak in ad:
        ad[ak]["agent"].start(ad[ak]["params"],e,mode="forkserver");
    st=time.time();
    for i in tqdm.tqdm(range(100)):
        aug_data=qd["dq"].get();
    et=time.time();
    for ak in ad:
        ad[ak]["agent"].stop();

    print(et-st);

    cv2.imshow("meow",aug_data["image"][0]);
    cv2.imshow("meow2", aug_data["image"][-1]);
    cv2.waitKey(0);

    pass;

