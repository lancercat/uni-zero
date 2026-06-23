import os.path

from osocr_tasks.ds_paths import get_cvpr16, get_nips14, get_iiit5k, get_SVT, get_cute, \
    get_IC03_867, get_IC13_1015, get_IC15_2077, get_SVTP
from osocrNG.data_utils.aug.determinstic_aug import augment_and_padding_agent
from osocrNG.data_utils.data_agents.multilmdb_agent import neko_multilmdb_fetching_agent
from osocrNG.data_utils.data_agents.multilmdb_dispatching_agent import neko_balance_fetching_and_dispatching_agent
from osocrNG.data_utils.indexer.multi_lmdb_indexer import neko_multi_lmdb_enumerator_rand_seed
from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder


def get_cvpr_2016_holder(dataroot):
    return neko_lmdb_holder({"root":get_cvpr16(dataroot)});
def get_nips_2014_holder(dataroot):
    return neko_lmdb_holder({"root":get_nips14(dataroot)});

# To make the process determinstic, please use only one loader to populate on queue
def get_mjst_v1(dataroot):
    holder=neko_named_multi_source_holder(
        {
            "sources":["CVPR2016","NIPS2014"],
            "sourced":{
                "CVPR2016": get_cvpr_2016_holder(dataroot),
                "NIPS2014": get_nips_2014_holder(dataroot),
            }
        }
    )
    sources=["CVPR2016", "NIPS2014"];
    loader=neko_multi_lmdb_enumerator_rand_seed(
        {
            "sources": sources,
            "ratio": {1,1},
            "lengths":[holder.sourced[k].nSamples for k in sources],
            "src_seed":9,
            "idx_seed":9,
        }
    )
    agent=neko_multilmdb_fetching_agent(
        {"sources": holder,"indexer":loader}
    )
    return agent;

def get_eng_test_v1(dataroot,v2h=-9):
    FNS=[get_iiit5k,get_SVT,get_cute,get_IC03_867,get_IC13_1015,get_IC15_2077,get_SVTP];
    NMS=["IIIT5k","SVT","CUTE","IC03","IC13","IC15","SVTP"];
    metadict={
        "EN":
              {"meta_path": os.path.join(dataroot, "dicts", "dab62cased.pt"),
               "case_sensitive": False,
               "has_unk": False
               }
              };
    datadict={
    };
    for d,n in zip(FNS,NMS):
        datadict[n]=neko_lmdb_holder({"root": d(dataroot),"vert_to_hori":v2h})

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


def get_en_v1_test(dataroot):
    pass;


def get_mjst_v1H(dataroot,adict):
    holder=neko_named_multi_source_holder(
        {
            "sources":["CVPR2016","NIPS2014"],
            "sourced":{
                "CVPR2016": get_cvpr_2016_holder(dataroot),
                "NIPS2014": get_nips_2014_holder(dataroot),
            }
        }
    )

    hydra_cfg={
        "sources": holder,
        "ancidx_path": os.path.join(dataroot, "anchors", "mjst-"+adict["profile_name"]+".pt"),
        "anchor_cfg": adict,
    }

    agent=neko_balance_fetching_and_dispatching_agent(
        hydra_cfg
    )
    return agent;


def index_mjst_v1H(dataroot,adict):
    holder=neko_named_multi_source_holder(
        {
            "sources":["CVPR2016","NIPS2014"],
            "sourced":{
                "CVPR2016": get_cvpr_2016_holder(dataroot),
                "NIPS2014": get_nips_2014_holder(dataroot),
            }
        }
    )

    hydra_cfg={
        "sources": holder,
        "ancidx_path": os.path.join(dataroot, "anchors", "mjst-"+adict["profile_name"]+".pt"),
        "anchor_cfg": adict,
    }

    agent=neko_balance_fetching_and_dispatching_agent(
        hydra_cfg
    )



def get_trmeta(dsroot):
    return os.path.join(dsroot, "dicts", "dab62cased.pt");
def arm_mjst_hydra_v1(agent_dict,qdict,params):
    qmap={};
    qname=params["data_queue_name"];

    ks = list(params["adict"]["names"]);

    for k in ks:
        qmap[k]=qname+"raw_"+k;

    da = get_mjst_v1H(params["dataroot"],params["adict"]);
    agent_dict["data_agent"]={
        "agent":da,
        "params":{
            "inputs":[],
            "outputs":ks,
            "remapping":{
                "queues":qmap,
                "assets": {
                }
            }
        }
    }
    for k in ks:
        aa = augment_and_padding_agent({
            "width": params["adict"][k]["target_size"][0],
            "height": params["adict"][k]["target_size"][1],
            "beacon_h":params["adict"][k]["beacon_size_wh"][1],
            "beacon_w": params["adict"][k]["beacon_size_wh"][0],
            "batch_size": params["adict"][k]["batch_size"],
        });
        agent_dict[k+"_data_augment"] = {
            "agent": aa,
            "params": {
                "inputs": ["raw_data"],
                "outputs": ["aligned_data"],
                "remapping": {
                    "queues": {
                        "raw_data": qname+"raw_"+k,
                        "aligned_data": qname+k,
                    },
                    "assets": {
                    }
                }
            }
        }
        qdict[qname+"raw_"+k]=None;
        qdict[qname+k] = None;

    return agent_dict,qdict;



