from osocrNG.data_utils.data_agents.sync_loaders.sync_oredered_single_lmdb import neko_single_im_text_sync_ordered_fetching_agent
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.abstract_data_factory import abstract_data_repo_mixin
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.names import P
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN


class single_ordered_lmdb_factory(abstract_data_repo_mixin):
    PARAM_root="dataroot";
    PARAM_force_grey="force_grey"
    PARAM_batch_size="batch_size";
    @classmethod
    def arm_disldr(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, P(data_prefix, "disk_ldr"));
        force_grey=neko_get_arg(cls.PARAM_force_grey,params,False);
        data_root=params[cls.PARAM_root];
        acfg = awa.append_agent_to_cfg(
            acfg, P(aprfx, "load_data"),
            neko_single_im_text_sync_ordered_fetching_agent.get_agtcfg(
                VN.SAM_ORIG(data_prefix),
                VN.IM_raw(data_prefix),
                VN.UTF(data_prefix),
                params[cls.PARAM_batch_size],
                "test",data_root,force_grey
            )
        );
        return acfg

if __name__ == '__main__':
    ac=awa.empty();
    ac=single_ordered_lmdb_factory.arm_data_repo(
        ac,{single_ordered_lmdb_factory.DATA_PRFX:"NepData"},{},{single_ordered_lmdb_factory.AGT_PRFX:"testingldr"},{single_ordered_lmdb_factory.PARAM_root:"/home/lasercat/ssddata/mlttrjp_hv/",single_ordered_lmdb_factory.PARAM_batch_size:32}
    );
    a=awa.make(ac);
    da=a.get_agt_at(0);
    from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
    e=neko_environment();
    for i in range(da.len()):
        ws=neko_workspace();
        a.take_action(ws,e);
        pass;

