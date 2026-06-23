# this just load anything.
# sampling is sampler's job.
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from osocrNG.data_utils.common_data_presets_mk4.presets.names import P
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.abstract_data_factory import abstract_data_repo_mixin

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile,neko_training_load_balance_profile,neko_db_profile


from osocrNG.data_utils.data_agents.sync_loaders.sync_scale_balanced import neko_im_text_sync_balance_fetching_and_mixing_mk2_agent
# from neko_sdk.semins.lmdb_utils.sync_panoptic_loader import neko_panoptic_sync_balance_fetching_and_mixing_agent
class im_text_multisource_lmdb_scale_balanced_fetch(abstract_data_repo_mixin):
    PARAM_dataloader_profile="data_loader_profile"
    PARAM_SEED="seed";

    @classmethod
    def arm_disldr(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, P(data_prefix, "disk_ldr"));
        seed = neko_get_arg(cls.PARAM_SEED, params, 9);
        profile = neko_get_arg(cls.PARAM_dataloader_profile, params);
        ancidx_path = profile[neko_datastream_profile.LDRIDX];
        dbcfg=profile[neko_datastream_profile.DB];
        dcfg = dbcfg[neko_db_profile.SRCS];
        balcfg = profile[neko_datastream_profile.BALCFG];
        force_grey=dbcfg[neko_db_profile.FORCE_GRAY];
        anccfg = {"names":[]};
        for k in balcfg[neko_training_load_balance_profile.TEMPLATE_NAMES]:
            anccfg["names"].append(k);
            anccfg[k]=balcfg[neko_training_load_balance_profile.TEMPLATES][k];

        data_names = list(dcfg.keys());
        data_roots = [dcfg[i] for i in data_names];
        acfg = awa.append_agent_to_cfg(
            acfg, P(aprfx, "load_data"),
            neko_im_text_sync_balance_fetching_and_mixing_mk2_agent.get_agtcfg(
                VN.SAM_ORIG(data_prefix),
                VN.IM_raw(data_prefix),
                VN.UTF(data_prefix),  # batch size is contra
                anccfg, ancidx_path, data_names,
                data_roots, seed=seed,force_grey=force_grey
            )
        );
        return acfg;
