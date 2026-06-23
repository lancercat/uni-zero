
import os.path
import time

import cv2
import tqdm

from osocrNG.data_utils.common_data_presets.dspathsNG import get_mltkr_path,get_mltjphv_path,get_mltjp_path
from osocrNG.data_utils.neko_imageset_holder import neko_image_holder
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.athena.common.analyze_folder import bootstrap_folder
from osocrNG.data_utils.common_data_presets_mk3.factory import abstract_mk3_data_factory
from neko_sdk.neko_framework_NG.data.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent_abinet,augment_agent_abinet_zeropadding,augment_agent_abinet_fixed
from osocrNG.data_utils.common_data_presets_mk5.eval_presets.moostr_ostr import moostr_ostr_nomask
from osocrNG.data_utils.common_data_presets_mk3.controlled_synth import controlled_synth
from osocrNG.data_utils.common_data_presets_mk3.uncontrolled_synth import uncontrolled_synth

def get_chs_training_metav2(root):
    return os.path.join(root, "dictsv2", "dab3791MC");

# using factory class to avoid wild strings.
# you can still use wild strings to call the factory,
# but now you can not to.
class moostr_mk3_data_factory(abstract_mk3_data_factory):

    @classmethod
    def get_osocr_test_gosr(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=moostr_ostr_nomask.arm_osocr_test_jpnhv_gosr(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_all(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=moostr_ostr_nomask.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict,dataroot,v2h);

        meta_dict, data_dict, test_dict=moostr_ostr_nomask.arm_osocr_test_jpnhv_full(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=moostr_ostr_nomask.arm_osocr_test_jpn_full(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_310(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_jpnhv_gzsl(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        # meta_dict, data_dict, test_dict=cls.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };

    @classmethod
    def get_osocr_test_310medoldsynth(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_jpnhv_gzsl(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_310medsynth(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_jpnhv_gzsl(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict = moostr_ostr_nomask.arm_osocr_test_kr_full(meta_dict, data_dict, test_dict, dataroot, v2h);
        meta_dict, data_dict, test_dict=controlled_synth.arm_synth_test_any(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_310sl(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_heavy_synth(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_kr_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_3755_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_jpn_heavy(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_mk3_test_adhoc(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict = {}, {}, {};
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_3755_adhoc(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_kr_adhoc(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict = uncontrolled_synth.arm_synth_test_jpn_adhoc(meta_dict, data_dict, test_dict, dataroot, v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };

    @classmethod
    def get_osocr_test_allsync(cls,dataroot, v2h=-9,pfx=""):
        meta_dict, data_dict, test_dict={},{},{};
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_3755(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_kr(meta_dict, data_dict, test_dict,dataroot,v2h);
        meta_dict, data_dict, test_dict=uncontrolled_synth.arm_synth_test_jpn(meta_dict, data_dict, test_dict,dataroot,v2h);

        return {
            "meta": meta_dict,
            "data": data_dict,
            "tests": test_dict
        };
    @classmethod
    def get_osocr_test_jpn_hv(cls,dataroot, v2h=-9,pfx=""):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 }
        };
        datadict = {
        };
        for d, n in zip(FNS, NMS):
            datadict[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})

        test_dict = {};
        for d in datadict:
            for m in metadict:
                test_dict[d + "-gzsl"+pfx] = {
                    "data": d,
                    "meta": m,
                }
        return {
            "meta": metadict,
            "data": datadict,
            "tests": test_dict
        };
    @classmethod
    def get_moostr_v1(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["art", "mlt", "ctw", "rctw", "lsvt"],
                he.PARAM_sourced: {
                    "art": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen_NG'), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"moostr-");
    @classmethod
    def get_moostr_v1_mk3(cls,dataroot, anchor_dict,data_queue_name, vert_to_hori=-9):
        he=neko_named_multi_source_holder;
        holder = he(
            {
                he.PARAM_sources: ["art", "mlt", "ctw", "rctw", "lsvt"],
                he.PARAM_sourced: {
                    "art": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'artdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "mlt": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'mlttrchlat_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "ctw": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'ctwdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "rctw": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'rctwtrdb_seen_NG'), "vert_to_hori": vert_to_hori}),
                    "lsvt": neko_im_label_lmdb_holder(
                        {"root": os.path.join(dataroot, 'lsvtdb_seen_NG'), "vert_to_hori": vert_to_hori})
                }
            }
        );
        return cls.get_mk2_loader_agent(holder,dataroot,anchor_dict,data_queue_name,"moostr-");

    @classmethod
    def arm_moostr_v1(cls,agent_dict, qdict, params):
        da = cls.get_moostr_v1(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h,params,-9)
        );
        agent_dict,qdict=cls.arm_training_data(da,agent_dict,qdict,params);
        meta=get_chs_training_metav2(params[cls.PARAM_dataroot]);
        return agent_dict,qdict,meta;

    @classmethod
    def arm_moostr_v1_mk3(cls, agent_dict, qdict, params):
        da = cls.get_moostr_v1_mk3(
            params[cls.PARAM_dataroot],
            params[cls.PARAM_anchor_dict],
            params[cls.PARAM_preaug_data_queue_name],
            neko_get_arg(cls.PARAM_v2h, params, -9)
        );
        agent_dict, qdict = cls.arm_training_data(da, agent_dict, qdict, params);
        meta = get_chs_training_metav2(params[cls.PARAM_dataroot]);
        return agent_dict, qdict, meta;
    @classmethod
    def get_benchmark(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_plus(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310(data_root,-9);
        return trad,trqd,trm,tedd;

    @classmethod
    def get_mk3_benchmark_plus_medsynth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310medsynth(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_plus_medoldsynth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310medoldsynth(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_sl(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310sl(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_release(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_310(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_heavy_synth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_heavy_synth(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_adhoc(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_mk3_test_adhoc(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_testall(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_all(data_root,-9);
        return trad,trqd,trm,tedd;
    @classmethod
    def get_mk3_benchmark_testsynth(cls,data_root,anchor_dict,queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                              {
                                                  cls.PARAM_dataroot: data_root,
                                                  cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
                                                  cls.EXPORT_data_queue_name: queue_name,
                                                  cls.PARAM_anchor_dict: anchor_dict,
                                              }
                                              );
        tedd=cls.get_osocr_test_allsync(data_root,-9);
        return trad,trqd,trm,tedd;

    @classmethod
    def get_mk3_benchmark_testgosr(cls, data_root, anchor_dict, queue_name):
        trad, trqd = {}, {};

        trad, trqd, trm = cls.arm_moostr_v1_mk3(trad, trqd,
                                                {
                                                    cls.PARAM_dataroot: data_root,
                                                    cls.PARAM_preaug_data_queue_name: "preaug_" + queue_name,
                                                    cls.EXPORT_data_queue_name: queue_name,
                                                    cls.PARAM_anchor_dict: anchor_dict,
                                                }
                                                );
        tedd = cls.get_osocr_test_gosr(data_root, -9);
        return trad, trqd, trm, tedd;
    # @classmethod
    # def get_mk3_benchmark_w_moe(cls,data_root,anchor_dict,queue_name):
    #     trad, trqd = {}, {};
    #
    #     trad, trqd, trm = cls.arm_moostr_v1_mk2(trad, trqd,
    #                                           {
    #                                               cls.PARAM_dataroot: data_root,
    #                                               cls.PARAM_preaug_data_queue_name: "preaug_"+queue_name,
    #                                               cls.EXPORT_data_queue_name: queue_name,
    #                                               cls.PARAM_anchor_dict: anchor_dict,
    #                                           }
    #                                           );
    #     tedd=cls.get_osocr_test_jpn_hv(data_root,-9);
    #     teddM=cls.get_osocr_test_jpn_hv(data_root,-9,"-MoE");
    #
    #     return trad,trqd,trm,tedd,teddM;


class moostr_mk3_data_factoryAA(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet;

class moostr_mk3_data_factoryAAZ(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_zeropadding;

class moostr_mk3_data_factoryAAF(moostr_mk3_data_factory):
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent_abinet_fixed;


# To make the process determinstic, please use only one loader to populate on queue





def get_osocr_test_jpn_hv_full(dataroot,v2h=-9):
    FNS = [get_mltjphv_path];
    NMS = ["JPNHV"];
    metadict = {
        "JPNHV-GZSL":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv"),
             "case_sensitive": False,
             "has_unk": False
             # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
             # For these test dss we need the model to pretend not seeing these characters
             },
        "JPNHV-OSR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_osr"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-GOSR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_nohirakata"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-OSTR":
            {"meta_path": os.path.join(dataroot, "dictsv2", "dabjpmlthv_kanjix"),
             "case_sensitive": False,
             "has_unk": True,
            }
    };
    datadict = {
    };
    for d, n in zip(FNS, NMS):
        datadict[n] = neko_im_label_lmdb_holder({"root": d(dataroot),"vert_to_hori":v2h})

    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d + "-"+m] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    };


def get_osocr_test_image_based(data_root,dst,lang,v2h=-9):
    files, ptfile, sfolder, dfolder=bootstrap_folder(data_root,dst,lang,"*.*");
    metadict = {
        "generic":
            {"meta_path": ptfile,
             "case_sensitive": True,
             "has_unk": False
             }
    };
    datadict={
        "generic":neko_image_holder({"files":files,"vert_to_hori":v2h})
    }
    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    },dfolder;


if __name__ == '__main__':
    from neko_sdk.neko_framework_NG.workspace import neko_environment

    ad,qd={},{};
    from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1v_6_05 as two_hori_main
    anchors=two_hori_main();

    de=moostr_mk3_data_factoryAA;
    ad,qd,meta=de.arm_moostr_v1(ad,qd,
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
        cv2.imshow("meow", aug_data["image"][0]);
        cv2.imshow("meow2", aug_data["image"][-1]);
        cv2.waitKey(0);

    et=time.time();
    for ak in ad:
        ad[ak]["agent"].stop();

    print(et-st);

    pass;
