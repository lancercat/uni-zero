
import os.path


from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.data_utils.common_data_presets_mk3.factory import abstract_mk3_data_factory
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocr_tasks.ds_paths import ( get_istr_bengali_test,get_istr_bengali_val,get_istr_gujarati_train,get_istr_hindi_test,get_istr_hindi_val,get_istr_kannada_train,get_istr_malayalam_test,get_istr_malayalam_val,get_istr_marathi_train,get_istr_punjabi_test,get_istr_punjabi_val,get_istr_tamil_train,get_istr_telugu_test,get_istr_telugu_val,get_istr_bengali_train,get_istr_gujarati_test,get_istr_gujarati_val,get_istr_hindi_train,get_istr_kannada_test,get_istr_kannada_val,get_istr_malayalam_train,get_istr_marathi_test,get_istr_marathi_val,get_istr_punjabi_train,get_istr_tamil_test,get_istr_tamil_val,get_istr_telugu_train
)
from osocr_tasks.ds_paths import get_hhd_test,get_hhd_test_1800ad,get_hhd_train
from osocrNG.data_utils.common_data_presets.dspathsNG import get_mltkr_path,get_mltjphv_path,get_mltjp_path,get_mltkr_path
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent,augment_agent_abinet,augment_agent_abinet_fixed

GRPS_te={
"bengali": [get_istr_bengali_test],
"gujarati":[get_istr_gujarati_test],
"hindi":[get_istr_hindi_test],
"kannada":[ get_istr_kannada_test],
"malayalam":[get_istr_malayalam_test],
"marathi":[get_istr_marathi_test],
"punjabi":[get_istr_punjabi_test],
"tamil":[get_istr_tamil_test],
"telugu":[get_istr_telugu_test],
"ethopian":[get_hhd_test,"dab_ethopic"],
"ethopian_18th":[get_hhd_test_1800ad,"dab_ethopic"],
}

# but now you can not to.
class moostrHE_mk3_data_factoryAAF(abstract_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_fixed;
    @classmethod
    def arm_osocr_test_jpnhv_gzsl(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict

    @classmethod
    def arm_osocr_test_jpnhv_full(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 },
            # "JPNHV-OSR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_osr"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      },
            # "JPNHV-GOSR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_nohirakata"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      },
            # "JPNHV-OSTR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_kanji"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict

    @classmethod
    def arm_osocr_test_jpn_full(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjp_path];
        NMS = ["JPN"];
        metadict_ = {
            "JPN-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlt"),
                 "case_sensitive": False,
                 "has_unk": False,
                 },
            # "JPN-OSR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_osr"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      },
            # "JPN-GOSR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_nohirakata"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      },
            # "JPN-OSTR":
            #     {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmltch_kanji"),
            #      "case_sensitive": False,
            #      "has_unk": True,
            #      }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];
        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_osocr_test_kr_full(cls,meta_dict,data_dict,test_dict,dataroot,v2h=-9):
        FNS = [get_mltkr_path];
        NMS = ["KR"];
        metadict_ = {
            "KR-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk":False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not  in meta_dict);
            meta_dict[m]=metadict_[m];
        data_dict_={}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n]=data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict,data_dict,test_dict
    @classmethod
    def arm_osocr_test_wkey(cls,key, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        if(len(GRPS_te[key])==1):
            FNS = GRPS_te[key];
            meta="dab_istr_"+key;
        else:
            FNS= [GRPS_te[key][0]];
            meta=GRPS_te[key][1];
        NMS = [key];
        metadict_ = {
            key+"-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", meta),
                 "case_sensitive": False,
                 "has_unk": False,
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];
        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict

    @classmethod
    def get_moosterHE_test_all(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);

        meta_dict, data_dict, test_dict = cls.arm_osocr_test_jpnhv_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = cls.arm_osocr_test_jpn_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        for k in GRPS_te:
            meta_dict, data_dict, test_dict=cls.arm_osocr_test_wkey(k,meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_moosterHE_v1_mk3(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["hindi", "kannada", "malayalam", "marathi", "punjabi","tamil","telugu",
                                   "art", "mlt", "ctw", "rctw", "lsvt","ethopian",
                                   ],
                he.PARAM_sourced: {
                    "art": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "hindi": neko_lmdb_holder(
                        {"root":get_istr_hindi_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "kannada": neko_lmdb_holder(
                        {"root": get_istr_kannada_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "malayalam": neko_lmdb_holder(
                        {"root": get_istr_malayalam_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "marathi": neko_lmdb_holder(
                        {"root": get_istr_marathi_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "punjabi": neko_lmdb_holder(
                        {"root": get_istr_punjabi_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "tamil": neko_lmdb_holder(
                        {"root": get_istr_tamil_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "telugu": neko_lmdb_holder(
                        {"root": get_istr_telugu_train(dataroot), "vert_to_hori": vert_to_hori}),
                    "ethopian": neko_lmdb_holder(
                        {"root": get_hhd_train(dataroot), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"moostrHE-");

    @classmethod
    def arm_moosterHE_v1_mk3(cls, agent_dict, qdict, params):
        da = cls.get_moosterHE_v1_mk3(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h, params, -9)
        );
        agent_dict, qdict = cls.arm_training_data(da, agent_dict, qdict, params);
        meta = os.path.join(params[cls.PARAM_dataroot],"dictsv2","dab_moostrHE");
        return agent_dict, qdict, meta;
    @classmethod
    def get_mk3_benchmark_plus(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moosterHE_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_moosterHE_test_all(data_root,-9);
        return trad,trqd,trm,tedd;
