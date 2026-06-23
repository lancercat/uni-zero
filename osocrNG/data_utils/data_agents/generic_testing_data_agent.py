import random

import torch
import tqdm

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent_gen1


# Yeah this thing can block, so only usable for testing, or training when you do not balance the anchors.

class neko_generic_testing_data_agent(neko_abstract_async_agent_gen1):
    def setup(this,param):
        this.datasource=param["sources"];
        this.indexer=param["indexer"];
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
