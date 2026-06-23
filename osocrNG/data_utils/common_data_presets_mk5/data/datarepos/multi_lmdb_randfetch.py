from osocrNG.data_utils.data_agents.sync_loaders.sync_multi_lmdb import neko_multilmdb_im_text_sync_fetching_agent

# this just load anything.
# sampling is sampler's job.
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from osocrNG.data_utils.common_data_presets_mk4.presets.names import P
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.abstract_data_factory import abstract_data_repo

from neko_sdk.cfgtool.argsparse import neko_get_arg

class multisource_lmdb_randfetch(abstract_data_repo):
    PARAM_data_dict="datadict";
    PARAM_batch_size="batch_size";
    @classmethod
    def arm_disldr(this, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[this.DATA_PRFX];
        aprfx = neko_get_arg(this.AGT_PRFX, agt_prfxs, P(data_prefix, "disk_ldr"));
        batch_size = neko_get_arg(this.PARAM_batch_size, params, 32);
        seed = neko_get_arg(this.PARAM_SEED, params, 9);
        data_names = list(params[this.PARAM_data_dict].keys());
        data_roots = [params[this.PARAM_data_dict][i] for i in data_names];
        acfg = awa.append_agent_to_cfg(
            acfg, P(aprfx, "load_data"),
            neko_multilmdb_im_text_sync_fetching_agent.get_agtcfg(
                VN.IM_raw(data_prefix),
                VN.UTF(data_prefix), batch_size,
                data_names,
                data_roots, seed=seed
            )
        );
        return acfg