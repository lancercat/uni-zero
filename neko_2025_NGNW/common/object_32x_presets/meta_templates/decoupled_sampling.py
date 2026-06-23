from neko_sdk.cfgtool.argsparse import neko_get_arg

from osocrNG.data_utils.common_data_presets_mk5.meta.samplers.text_based_samper import multi_text_based_meta_sampler

from neko_sdk.neko_framework_NG.names import P
from osocrNG.data_utils.common_data_presets_mk5.meta.tdict_holders.pt_tdict_loader import single_trivial_meta_repo
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory

# we only keep one single image modality,
# and the sampling is done by a selected data
# this very implementation does not modify tdict
class neko_meta_ldr_agt_factory_core(virtual_agt_factory):
    PARAM_meta_root="meta_root"; # this just loads tdict/utf and samples them--- you need to figure out sinfos down the road
    HOLDRF=single_trivial_meta_repo;
    def setup(this):
        this.holder_factory=this.HOLDRF();
    def __init__(this):
        this.setup();
    def arm_meta_ldr_agt(this, agtcfg, agt_prfx, raw_meta_prfx,meta_root):
        agtcfg = this.holder_factory.arm_agts(agtcfg, {this.holder_factory.META_PRFX: raw_meta_prfx}, {},
                                              {this.holder_factory.AGT_PRFX: agt_prfx},
                                              {this.holder_factory.PARAM_meta_root: meta_root});
        return agtcfg



class training_sampled_meta_agt_factory(neko_meta_ldr_agt_factory_core):
    DATA_PRFX_refdata="refdata_prfx";# this calls
    PARAM_SEED = "seed";
    PARAM_max_proto_capacity="max_proto_capacity";
    PARAM_frac="frac";
    SAM=multi_text_based_meta_sampler; # eventually will make them agent factories too, but bear with me
    DATA_PRFX_unsampled="unsampled_meta_prfx";

    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        sampled_meta_prfx=data_prfx_dict[this.DATA_PRFX_local];
        ref_data_prfx = data_prfx_dict[this.DATA_PRFX_refdata];
        raw_meta_prfx=data_prfx_dict[this.DATA_PRFX_unsampled]; # as you have noticed the endpoint is unsampled is not the endpoint here.
        meta_root = params[this.PARAM_meta_root];
        frac = params[this.PARAM_frac];
        capacity = params[this.PARAM_max_proto_capacity];
        seed=params[this.PARAM_SEED];
        agtcfg=this.arm_meta_ldr_agt(agtcfg,agt_prfx,raw_meta_prfx,meta_root);
        agtcfg=this.SAM.arm_sampler(agtcfg,{this.SAM.META_PRFX:raw_meta_prfx,this.SAM.REF_DATA_PRFXS:ref_data_prfx,this.SAM.SAMPLED_META_DATA_PRFX:sampled_meta_prfx},
                                    {},{this.SAM.AGT_PRFX:agt_prfx},{this.SAM.PARAM_seed:seed,this.SAM.PARAM_frac:frac,this.SAM.PARAM_max_proto_capacity:capacity});
        return agtcfg

class testing_meta_agt_factory(neko_meta_ldr_agt_factory_core):
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        raw_meta_prfx = data_prfx_dict[this.DATA_PRFX_local];
        meta_root = params[this.PARAM_meta_root];
        return this.arm_meta_ldr_agt(agtcfg, agt_prfx, raw_meta_prfx, meta_root);

