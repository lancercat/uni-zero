from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent_gen1


class neko_multilmdb_fetching_agent(neko_abstract_async_agent_gen1):
    def setup(this,param):
        this.datasource=param["sources"];
        this.indexer=param["indexer"];
    def action_loop(this):
        while this.status==this.STATUS_running:
            data=this.datasource.fetch_item(this.indexer.__next__());
            if(data is not None):
                this.environment.queue_dict["data"].put(data);
