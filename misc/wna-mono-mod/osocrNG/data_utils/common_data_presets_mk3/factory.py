from neko_sdk.neko_framework_NG.UAE.neko_mission_agent import neko_test_mission_agent as TN
from osocrNG.data_utils.data_agents.multilmdb_anchor_balanced_mixed_agent import (
    neko_balance_fetching_and_mixing_agent,neko_balance_fetching_and_mixing_agent_mk2)
from osocrNG.data_utils.aug.determinstic_aug_mk2 import augment_agent,augment_agent_abinet
import os

class abstract_mk3_data_factory:
    EXPORT_data_queue_name="data_queue_name";
    PARAM_preaug_data_queue_name="preaug_data_queue_name";
    PARAM_dataroot="dataroot";
    PARAM_anchor_dict="adict";
    PARAM_v2h="vert_to_hori"
    ANAME_data_agent="data_agent";
    ANAME_data_augment_agent="data_augment";
    ANAME_data_dispatching_agent="data_predispaching";
    @classmethod
    def AUG_ENGINE(cls):
        return augment_agent;
    @classmethod
    def arm_training_data(cls,da,agent_dict,qdict,params):
        qname = params[cls.EXPORT_data_queue_name];
        pre_aug_qname = params[cls.PARAM_preaug_data_queue_name];
        # Since this one mixes queues, no need for remapping
        # Remind you remapping is a process
        # so that a same agent class can be registered to handle different data streams.
        ae=neko_balance_fetching_and_mixing_agent
        agent_dict[cls.ANAME_data_agent]={
            "agent":da,
            "params":{
                ae.PARAM_INPUT_QUEUES:[],
                ae.PARAM_OUTPUT_QUEUES:[pre_aug_qname],
            }
        }
        #yes you can have a second, per-anchor beacon,
        # using collator if you do need one.
        # It just turtles down.
        ae=cls.AUG_ENGINE();
        aa = ae({
            ae.PARAM_seed:9,
            ae.PARAM_augmenter_workers:9,
            ae.EXPORT_Q:qname,
            ae.IMOPRT_Q:pre_aug_qname
        });
        agent_dict[cls.ANAME_data_augment_agent] = {
            "agent": aa,
            "params": {
                "inputs": [pre_aug_qname],
                "outputs": [qname],
            }
        }
        qdict[qname]=None;
        qdict[pre_aug_qname]=None;
        return agent_dict,qdict;
    @classmethod
    def get_loader_agent(cls,holder,dataroot,anchor_dict,data_queue_name,prfx):
        ae = neko_balance_fetching_and_mixing_agent;
        hydra_cfg = {
            ae.PARAM_sources: holder,
            ae.PARAM_ancidx_path: os.path.join(dataroot, "anchors", prfx + anchor_dict["profile_name"] + ".pt"),
            ae.PARAM_anchor_cfg: anchor_dict,
            ae.EXPORT_Q: data_queue_name
        };
        agent = neko_balance_fetching_and_mixing_agent(
            hydra_cfg
        )
        return agent;

    # this one comes with more fine-grained anchor controlling,
    # however, does not take older anchor configs.
    @classmethod
    def get_mk2_loader_agent(cls, holder, dataroot, anchor_dict, data_queue_name, prfx):
        ae = neko_balance_fetching_and_mixing_agent_mk2;
        hydra_cfg = {
            ae.PARAM_sources: holder,
            ae.PARAM_ancidx_path: os.path.join(dataroot, "anchors", prfx +anchor_dict["profile_name"] + ".pt"),
            ae.PARAM_anchor_cfg: anchor_dict,
            ae.EXPORT_Q: data_queue_name
        };
        agent = neko_balance_fetching_and_mixing_agent_mk2(
            hydra_cfg
        )
        return agent;