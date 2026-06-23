from neko_2024_NGNW.common.heads_mk3.danhead import dan_head_factory_mk3
from neko_2024_NGNW.common.heads_mk3.danhead_alter_rewards import dan_head_factory_mk3_nedmix


# string based head config api.
class chaos_head:
    def __init__(this):
        this.HEADS={
            "danish":dan_head_factory_mk3(),
            "dan-nedmix": dan_head_factory_mk3_nedmix(),
        }
    # well some auto regression heads need gt for training
    def get_head_training_agent_by_string(this,prefix,name,head_type,params):
        return this.HEADS[head_type].get_head_training_agent_by_string(prefix,name,params);
    def get_head_testing_agent_by_string(this,prefix,name,head_type,params):
        return this.HEADS[head_type].get_head_testing_agent_by_string(prefix,name,params);
    def get_head_mod_by_string(this, modcfg,prefix,name, head_type,params,opt_params):
        return this.HEADS[head_type].get_head_mod_by_string(modcfg,prefix,name,params,opt_params);
    def get_head_training_extra_mod_by_string(this,modcfg,prefix,name,head_type,params,opt_params):
        return this.HEADS[head_type].get_head_training_extra_mod_by_string(modcfg,prefix,name,params,opt_params);


