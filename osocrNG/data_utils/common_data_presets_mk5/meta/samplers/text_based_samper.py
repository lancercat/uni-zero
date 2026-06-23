import tqdm

from neko_sdk.cfgtool.argsparse import neko_get_arg
# this just load anything.
# sampling is sampler's job.

from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.neko_framework_NG.libmeta.agents.tdict_repo import neko_mksrted_utf
from neko_sdk.neko_framework_NG.libmeta.agents.agentic_sampler.neko_agentic_sampler_mk1 import neko_agentic_label_grounded_sampler,neko_agentic_label_grounded_sampler_multi
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

class single_text_based_meta_sampler(neko_virtual_factory_mk5):
    # load lmdb, tdict, meta
    META_PRFX="meta_prefix";
    REF_DATA_PRFX="ref_dataprfx";
    META_DATA_PRFX="meta_data_prefix";
    PARAM_max_proto_capacity="max_proto_capacity";
    PARAM_frac="frac";
    PARAM_seed="seed";
    @classmethod
    def arm_sampler(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        meta_prfx=neko_get_arg(cls.META_PRFX,data_prfxs);
        ref_data_prfx=neko_get_arg(cls.REF_DATA_PRFX,data_prfxs);
        meta_data_prfx=neko_get_arg(cls.META_DATA_PRFX,data_prfxs);
        agt_prfxs = neko_get_arg(cls.AGT_PRFX, agt_prfxs,meta_prfx);
        frac=neko_get_arg(cls.PARAM_frac,params,0.8);
        max_proto_capacity=neko_get_arg(cls.PARAM_max_proto_capacity,params,256);
        seed = neko_get_arg(cls.PARAM_seed, params, 9);

        acfg = awa.append_agent_to_cfg(acfg,  P(agt_prfxs,"sample_meta"), neko_agentic_label_grounded_sampler.get_agtcfg(
            VN.GT_TOK_UTF(ref_data_prfx),
            VN.TDICT(meta_prfx),
            VN.TDICT(meta_data_prfx),
            frac, max_proto_capacity, seed));
        acfg=awa.append_agent_to_cfg(acfg,P(agt_prfxs,"make_utf"),neko_mksrted_utf.get_agtcfg(
            VN.TDICT(meta_data_prfx),
            VN.UTF(meta_data_prfx)
        ));

        return acfg


class multi_text_based_meta_sampler(neko_virtual_factory_mk5):
    # load lmdb, tdict, meta
    META_PRFX="meta_prefix";
    REF_DATA_PRFXS="ref_dataprfxs";
    SAMPLED_META_DATA_PRFX="meta_data_prefix";
    PARAM_max_proto_capacity="max_proto_capacity";
    PARAM_frac="frac";
    PARAM_seed="seed";
    @classmethod
    def arm_sampler(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        meta_prfx=neko_get_arg(cls.META_PRFX,data_prfxs);
        ref_data_prfxs=neko_get_arg(cls.REF_DATA_PRFXS,data_prfxs);
        sampled_meta_prfx=neko_get_arg(cls.SAMPLED_META_DATA_PRFX,data_prfxs);
        agt_prfxs = neko_get_arg(cls.AGT_PRFX, agt_prfxs,meta_prfx);
        frac=neko_get_arg(cls.PARAM_frac,params,0.8);
        max_proto_capacity=neko_get_arg(cls.PARAM_max_proto_capacity,params,256);
        seed = neko_get_arg(cls.PARAM_seed, params, 9);

        acfg = awa.append_agent_to_cfg(acfg,  P(agt_prfxs,"sample_meta"), neko_agentic_label_grounded_sampler_multi.get_agtcfg(
            [VN.GT_TOK_UTF(data_prfx) for data_prfx in ref_data_prfxs],
            VN.TDICT(meta_prfx),
            VN.TDICT(sampled_meta_prfx),
            frac, max_proto_capacity, seed));
        acfg=awa.append_agent_to_cfg(acfg,P(agt_prfxs,"make_utf"),neko_mksrted_utf.get_agtcfg(
            VN.TDICT(sampled_meta_prfx),
            VN.UTF(sampled_meta_prfx)
        ));

        return acfg


