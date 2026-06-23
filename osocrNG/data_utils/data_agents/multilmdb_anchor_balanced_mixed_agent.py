import random

import torch
import tqdm
from multiprocessing import Queue as mpQueue
import multiprocessing

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent_gen1
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.data_utils.data_agents.multilmdb_dispatching_agent import neko_fetching_and_dispatching_servant
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.data_utils.raw_names import raw_data_item_names as RN

from neko_sdk.log import fatal,warn


class neko_balance_fetching_and_mixing_agent(neko_abstract_async_agent_gen1):
    PARAM_sources="sources";
    PARAM_ancidx_path="ancidx_path";
    PARAM_anchor_cfg="anchor_cfg";
    EXPORT_Q="output_queue_name";


    def make_anc_idx(this):
        this.ancidx={};
        for k in this.anchor_names:
            this.ancidx[k]=[];
        for idx in tqdm.tqdm(this.datasource.all_valid_indexes()):
            if(idx["descp"]["id"]%100==9):
                this.datasource.reset_txn(idx)
            data=this.datasource.fetch_item(idx);
            try:
                data[RN.IMAGE].tobytes();
            except:
                warn("hidden corrupted sample");
                continue

            if(data is not None):
                ratio = data[RN.IMAGE].width / data[RN.IMAGE].height;
                if(this.anchor_training_ratio_ranges is None):
                    for i in range(len(this.ratio_anchors)):
                        if (ratio > this.ratio_anchors[i]):
                            this.ancidx[this.anchor_names[i]].append(idx);
                            break;
                else:
                    for i in range(len(this.anchor_training_ratio_ranges)):
                        if(ratio>this.anchor_training_ratio_ranges[i][0] and ratio<=this.anchor_training_ratio_ranges[i][1]):
                            this.ancidx[this.anchor_names[i]].append(idx);
    def setmeta(this,param):
        this.datasource = param[this.PARAM_sources];
        this.ancidx_path=param[this.PARAM_ancidx_path];
        this.anchor_names = param[this.PARAM_anchor_cfg]["names"];

        this.ratio_anchors = [
            param[this.PARAM_anchor_cfg][k]["ratio"] for k  in this.anchor_names];
        this.maxT=[
            param[this.PARAM_anchor_cfg][k]["maxT"] for k in this.anchor_names];
        this.minT = [
            param[this.PARAM_anchor_cfg][k]["maxT"] for k in this.anchor_names];
        try:
            training_ranges = [
                param[this.PARAM_anchor_cfg][k]["training_ar_range"] for k in this.anchor_names];
        except:
            training_ranges = None;
        this.batch_sizes = {};
        for k in this.anchor_names:
            this.batch_sizes[k] = param[this.PARAM_anchor_cfg][k]["batch_size"];
        this.anchor_training_ratio_ranges=training_ranges;

    def setup(this,param):
        this.servants={};
        this.setmeta(param);


        this.export_q=neko_get_arg(this.EXPORT_Q,param);
        if ("indexer" not in param):
            try:
                print("loading anchor index",this.ancidx_path);
                this.ancidx=torch.load(this.ancidx_path,weights_only=False);
                print("anchor index loaded");

            except:
                print()
                this.make_anc_idx();
                torch.save(this.ancidx,this.ancidx_path);
        else:
            fatal("error");

    def start(this,mapping_param,environment,mode="forkserver"):
        this.status = this.STATUS_running;
        this.servants={};
        this.queue_dict={};
        for k in this.ancidx:
            if(len(this.ancidx[k])):
                this.queue_dict[k]=mpQueue(maxsize=9);
        for k in this.ancidx:
            if(len(this.ancidx[k])):

                this.servants[k]= neko_fetching_and_dispatching_servant(
                    {"datasource":this.datasource,
                     "seed":9,
                     "ancidx":this.ancidx[k],
                     "queue":this.queue_dict[k],
                     }
                );

        for k in this.servants:
            this.servants[k].start(None,None,mode);
        this.mount_environment(mapping_param, environment);
        if (mode == "forkserver"):
            this.worker = multiprocessing.Process(target=this.action_loop);
        else:
            this.worker = multiprocessing.get_context(mode).Process(target=this.action_loop);
        this.worker.start();
        pass;
    def action_loop(this):
        while this.status==this.STATUS_running:
            data=[]
            for k in this.queue_dict:
                bs=this.batch_sizes[k];
                for i in range(bs):
                    rd=this.queue_dict[k].get();
                    rd[RN.ANCHOR]=k; # if you want hint the reward with the anchor it comes from, here is your chance.
                    data.append(rd);
            this.environment.queue_dict[this.export_q].put(data);

    def stop(this):
        this.status=this.STATUS_stopped;
        for s in this.servants:
            this.servants[s].stop();
        if(this.worker is not None):
            this.worker.kill();
            this.worker=None;

    def stop_and_quit(this):
        for s in this.servants:
            s.stop_and_quit();
        exit(0);



# training range is good idea and all, but it should be defined on commander/router/or even expert side. Not on data loader for sure.

class neko_balance_fetching_and_mixing_agent_deprecated(neko_balance_fetching_and_mixing_agent):
    def make_anc_idx(this):
        this.ancidx={};
        for k in this.anchor_names:
            this.ancidx[k]=[];
        for idx in tqdm.tqdm(this.datasource.all_valid_indexes()):
            if("descp" in idx): # multi source
                if(idx["descp"]["id"]%100==9):
                    this.datasource.reset_txn(idx); # reset txn to avoid deadlock
            elif("id" in idx): # single source
                if (idx["id"] % 100 == 9):
                    this.datasource.reset_txn(idx);

            data=this.datasource.fetch_item(idx);
            try:
                data[RN.IMAGE].tobytes();
            except:
                warn("hidden corrupted sample");
                continue

            if(data is not None):
                ratio = data[RN.IMAGE].width / data[RN.IMAGE].height;
                txt=data[RN.LABEL];
                putcnt=0;
                for i in range(len(this.anchor_training_ratio_ranges)):
                    if (len(txt) > this.maxT[i] or len(txt)<this.minT[i]):
                        continue;
                    if(ratio<this.anchor_training_ratio_ranges[i][0] or ratio>this.anchor_training_ratio_ranges[i][1]):
                        continue;
                    putcnt+=1;
                    this.ancidx[this.anchor_names[i]].append(idx);
                if(putcnt==0):
                    print("we get an orphan:", idx,"txt:", txt);
    def setmeta(this,param):
        this.datasource = param[this.PARAM_sources];
        this.ancidx_path=param[this.PARAM_ancidx_path];
        this.anchor_names = param[this.PARAM_anchor_cfg]["names"];
        this.maxT=[
            param[this.PARAM_anchor_cfg][k]["maxT"] for k in this.anchor_names];
        this.minT = [
            param[this.PARAM_anchor_cfg][k]["minT"] for k in this.anchor_names];
        # Mk2 now enforces defining anchors with aspect ratio ranges.
        this.anchor_training_ratio_ranges = [
                param[this.PARAM_anchor_cfg][k]["training_ar_range"] for k in this.anchor_names];
        this.batch_sizes = {};
        for k in this.anchor_names:
            this.batch_sizes[k] = param[this.PARAM_anchor_cfg][k]["batch_size"];
