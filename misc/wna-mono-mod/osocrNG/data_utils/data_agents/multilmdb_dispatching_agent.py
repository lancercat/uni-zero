import random

import torch
import tqdm

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent
from osocrNG.data_utils.raw_names import raw_data_item_names as RN


# Yeah this thing can block, so only usable for testing, or training when you do not balance the anchors.

class neko_fetching_and_dispatching_agent(neko_abstract_async_agent):
    def setup(this,param):
        this.datasource=param["sources"];
        this.indexer=param["indexer"];
        this.ratio_anchors=param["ratio_anchors"];
        this.anchor_names=param["anchor_names"];
    def action_loop(this):
        while this.status==this.STATUS_running:
            data=this.datasource.fetch_item(this.indexer.__next__());
            if(data is not None):
                w,h=data["image"].width,data["image"].height;
                ratio=w/h;
                for i in range(len(this.ratio_anchors)):
                    if(ratio>this.ratio_anchors[i]):
                        this.environment.queue_dict[this.anchor_names[i]].put(data);
                        break;

class neko_fetching_and_dispatching_servant(neko_abstract_async_agent):
    def setup(this,param):
        this.datasource=param["datasource"];
        this.rng=random.Random(param["seed"]);
        this.ancidx=param["ancidx"];
        this.queue=param["queue"];
    def fetch_src(this):
        itemdscp=this.rng.choice(this.ancidx);
        data = this.datasource.fetch_item(itemdscp);
        if(data is None):
            return this.fetch_src();
        w, h = data[RN.IMAGE].width, data[RN.IMAGE].height;
        if (min(w, h) < 5):
            return this.fetch_src();
        data[RN.UID]=itemdscp;
        # print("putting from", i);
        this.queue.put(data);
    def action_loop(this):
        while this.status==this.STATUS_running:
            this.fetch_src();
    def mount_environment(this,param,environment):
        pass;

class neko_balance_fetching_and_dispatching_agent(neko_abstract_async_agent):
    def make_anc_idx(this):
        this.ancidx={};
        for k in this.anchor_names:
            this.ancidx[k]=[];
        for idx in tqdm.tqdm(this.datasource.all_valid_indexes()):
            if(idx["descp"]["id"]%100==9):
                this.datasource.reset_txn(idx)
            data=this.datasource.fetch_item(idx);
            try:
                data["image"].tobytes();
            except:
                print("hidden corrupted image");
                continue

            if(data is not None):
                ratio = data["image"].width / data["image"].height;
                if(this.anchor_training_ratio_ranges is None):
                    for i in range(len(this.ratio_anchors)):
                        if (ratio > this.ratio_anchors[i]):
                            this.ancidx[this.anchor_names[i]].append(idx);
                            break;
                else:
                    for i in range(len(this.anchor_training_ratio_ranges)):
                        if(ratio>this.anchor_training_ratio_ranges[i][0] and ratio<=this.anchor_training_ratio_ranges[i][1]):
                            this.ancidx[this.anchor_names[i]].append(idx);

    def setup(this,param):
        this.servants={};

        this.datasource = param["sources"];
        this.ancidx_path=param["ancidx_path"];
        this.anchor_names = param["anchor_cfg"]["names"];
        this.ratio_anchors = [
            param["anchor_cfg"][k]["ratio"] for k  in this.anchor_names];
        try:
            training_ranges=[
                param["anchor_cfg"][k]["training_range"] for k in this.anchor_names];
        except:
            training_ranges=None;
        this.anchor_training_ratio_ranges=training_ranges;

        if ("indexer" not in param):
            try:
                print("loading anchor index",param["ancidx_path"]);
                this.ancidx=torch.load(param["ancidx_path"]);
                print("anchor index loaded");

            except:
                print()
                this.make_anc_idx();
                torch.save(this.ancidx,param["ancidx_path"]);
        else:
            fatal("error");

    def start(this,mapping_param,environment,mode="forkserver"):
        this.stop();
        this.mount_environment(mapping_param,environment);
        this.servants={};
        for k in this.ancidx:
            this.servants[k]= neko_fetching_and_dispatching_servant(
                {"datasource":this.datasource,
                 "seed":9,
                 "ancidx":this.ancidx[k],
                 "queue":this.environment.queue_dict[k],
                 }
            );

        for k in this.ancidx:
            this.servants[k].start(None,None,mode);
        this.status="running";

        pass;



    def stop(this):
        for s in this.servants:
            this.servants[s].stop();
    def stop_and_quit(this):
        for s in this.servants:
            s.stop_and_quit();
        exit(0);

    def action_loop(this):
        fatal("error");
