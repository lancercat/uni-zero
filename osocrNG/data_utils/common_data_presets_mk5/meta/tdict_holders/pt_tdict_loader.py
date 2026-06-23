from neko_sdk.cfgtool.argsparse import neko_get_arg
# this just load anything.
# sampling is sampler's job.
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.libmeta.agents.tdict_repo import neko_agentic_tdict_repo_static
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa


class single_trivial_meta_repo(neko_virtual_factory_mk5):
    # load lmdb, tdict, meta
    PARAM_meta_root="meta_root";
    META_PRFX="meta_prefix";
    @classmethod
    def get_meta_repo(cls,  data_prfxs, mod_prfxs, agt_prfxs, params):
        meta_root=neko_get_arg(cls.PARAM_meta_root,params);
        meta_prfx=neko_get_arg(cls.META_PRFX,data_prfxs)
        return neko_agentic_tdict_repo_static.get_agtcfg(
            VN.TDICT_PATH(meta_prfx), VN.TDICT_HASH(meta_prfx),
            VN.TDICT(meta_prfx), VN.UTF(meta_prfx),
            meta_root
        );


    def arm_agts(this, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        meta_prfx=neko_get_arg(this.META_PRFX,data_prfxs)
        agt_prfxs = neko_get_arg(this.AGT_PRFX, agt_prfxs,meta_prfx);
        ac=this.get_meta_repo( data_prfxs, mod_prfxs, agt_prfxs, params);
        # the utf list is built anyways, its smol. if you want to do magic do it elsewhere
        acfg = awa.append_agent_to_cfg(acfg, P(agt_prfxs,"load_tdict"),ac);
        return acfg

if __name__ == '__main__':
    from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

    acfg=awa.empty();
    af=single_trivial_meta_repo;
    acfg=af.arm_agts(acfg,{af.META_PRFX:"nep"},{},{af.AGT_PRFX:"shaaarrrkkkk"},{af.PARAM_meta_root:
                                                                         "/home/lasercat/ssddata/dicts_v3/chs3755_en_uncased_digits/"})
    a=acfg["agent"](acfg["params"]);
    ws=neko_workspace();
    e=neko_environment();
    a.take_action(ws,e);
    pass;

