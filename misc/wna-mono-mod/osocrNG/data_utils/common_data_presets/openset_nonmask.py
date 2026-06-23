import os.path

from osocrNG.data_utils.common_data_presets.dspathsNG import get_mltjphv_path
from osocrNG.data_utils.aug.determinstic_aug import augment_and_padding_agent
from osocrNG.data_utils.data_agents.multilmdb_dispatching_agent import neko_balance_fetching_and_dispatching_agent
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocrNG.data_utils.neko_imageset_holder import neko_image_holder
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.athena.common.analyze_folder import bootstrap_folder

def get_mltjp_path(root):
    return os.path.join(root, "mlttrjp_hori");



# To make the process determinstic, please use only one loader to populate on queue

def get_chslat_v1(dataroot, adict,vert_to_hori=-9):
    holder = neko_named_multi_source_holder(
        {
            "sources": ["art", "mlt", "ctw", "rctw", "lsvt"],
            "sourced": {
                # "art": get_artNG_holder(dataroot),
                # "mlt": get_mltchNG_holder(dataroot),
                # "ctw": get_ctwNG_holder(dataroot),
                # "rctw": get_rctwNG_holder(dataroot),
                # "lsvt": get_lsvtNG_holder(dataroot), neko_lmdb_holder({"root": get_lsvtKNG_path(dataroot)})
                "art": neko_lmdb_holder({"root": os.path.join(dataroot,'artdb_seen_NG'),"vert_to_hori":vert_to_hori}),
                "mlt": neko_lmdb_holder({"root": os.path.join(dataroot,'mlttrchlat_seen_NG'),"vert_to_hori":vert_to_hori}),
                "ctw": neko_lmdb_holder({"root": os.path.join(dataroot,'ctwdb_seen_NG'),"vert_to_hori":vert_to_hori}),
                "rctw": neko_lmdb_holder({"root": os.path.join(dataroot,'rctwtrdb_seen_NG'),"vert_to_hori":vert_to_hori}),
                "lsvt": neko_lmdb_holder({"root": os.path.join(dataroot,'lsvtdb_seen_NG'),"vert_to_hori":vert_to_hori})
            }
        }
    );
    anchor_path=os.path.join(dataroot, "anchors", "chslat-" + adict["profile_name"] + ".pt");
    if(vert_to_hori>0):
        anchor_path=os.path.join(dataroot, "anchors", "chslat-" + adict["profile_name"] + str(vert_to_hori)+ ".pt")
    hydra_cfg = {
        "sources": holder,
        "ancidx_path": anchor_path,
        "anchor_cfg": adict,
    }

    agent = neko_balance_fetching_and_dispatching_agent(
        hydra_cfg
    )
    return agent


def arm_chslat_v1(agent_dict, qdict, params):
    qmap = {}
    qname = params["data_queue_name"]

    ks = list(params["adict"]["names"])
    vert_to_hori=neko_get_arg("vert_to_hori",params,-9);
    for k in ks:
        qmap[k] = qname + "raw_" + k
    da = get_chslat_v1(params["dataroot"], params["adict"],vert_to_hori);
    agent_dict["data_agent"] = {
        "agent": da,
        "params": {
            "inputs": [],
            "outputs": ks,
            "remapping": {
                "queues": qmap,
                "assets": {
                }
            }
        }
    }
    for k in ks:
        aa = augment_and_padding_agent({
            "width": params["adict"][k]["target_size"][0],
            "height": params["adict"][k]["target_size"][1],
            "beacon_h": params["adict"][k]["beacon_size_wh"][1],
            "beacon_w": params["adict"][k]["beacon_size_wh"][0],
            "batch_size": params["adict"][k]["batch_size"],
        });
        agent_dict[k + "_data_augment"] = {
            "agent": aa,
            "params": {
                "inputs": ["raw_data"],
                "outputs": ["aligned_data"],
                "remapping": {
                    "queues": {
                        "raw_data": qname + "raw_" + k,
                        "aligned_data": qname + k,
                    },
                    "assets": {
                    }
                }
            }
        }
        qdict[qname + "raw_" + k] = None;
        qdict[qname + k] = None;

    return agent_dict, qdict;


def get_osocr_test_oldjpn(dataroot):
    FNS = [get_mltjp_path];
    NMS = ["JPN"];
    metadict = {
        "JPN-GZSL":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlt.pt"),
             "case_sensitive": False,
             "has_unk":False,
             },
        "JPN-OSR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmltch_osr.pt"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPN-GOSR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmltch_nohirakata.pt"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPN-OSTR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmltch_kanji.pt"),
             "case_sensitive": False,
             "has_unk": True,
             }
    };
    datadict = {
    };
    for d, n in zip(FNS, NMS):
        datadict[n] = neko_lmdb_holder({"root": d(dataroot)})

    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d + "-" + m] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    };


def get_osocr_test_jpn_hv(dataroot,v2h=-9):
    FNS = [get_mltjphv_path];
    NMS = ["JPNHV"];
    metadict = {
        "JPNHV-GZSL":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlthv.pt"),
             "case_sensitive": False,
             "has_unk": False
             }
    };
    datadict = {
    };
    for d, n in zip(FNS, NMS):
        datadict[n] = neko_lmdb_holder({"root": d(dataroot),"vert_to_hori":v2h})

    test_dict = {};
    for d in datadict:
        for m in metadict:
            test_dict[d + "-gzsl"] = {
                "data": d,
                "meta": m,
            }
    return {
        "meta": metadict,
        "data": datadict,
        "tests": test_dict
    };

def get_osocr_test_jpn_hv_full(dataroot,v2h=-9):
    FNS = [get_mltjphv_path];
    NMS = ["JPNHV"];
    metadict = {
        "JPNHV-GZSL":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlthv.pt"),
             "case_sensitive": False,
             "has_unk": False
             # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
             # For these test dss we need the model to pretend not seeing these characters
             },
        "JPNHV-OSR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlthv_osr.pt"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-GOSR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlthv_nohirakata.pt"),
             "case_sensitive": False,
             "has_unk": True,
             },
        "JPNHV-OSTR":
            {"meta_path": os.path.join(dataroot, "dicts", "dabjpmlthv_kanji.pt"),
             "case_sensitive": False,
             "has_unk": True,
            }
    };
    datadict = {
    };
    for d, n in zip(FNS, NMS):
        datadict[n] = neko_lmdb_holder({"root": d(dataroot),"vert_to_hori":v2h})

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

def get_chs_training_meta(root):
    return os.path.join(root, "dicts", "dab3791MC.pt");
