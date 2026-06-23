from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_factory, virtual_mod_factory
from neko_sdk.CnC.situation_assess.neko_r45_assessment import neko_assess_r45_direct_se
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk1 import spatial_attention_NG_mk1
from neko_sdk.neko_framework_NG.bogog2_modules.featmap_to_feat import gen4_featmap_to_feat_abstract


# on the way to chunk it. eventually we want to get normalization and collation done on cpu in parallel.
# the gist is only embark on to gpu here to reduce wasting time waiting for normalization.
class neko_dedicated_res45(virtual_mod_factory):
    PARAM_outdim="outdim";
    PARAM_indim="indim";


    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        prfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_saveable(
            modcfg,neko_assess_r45_direct_se,{
                neko_assess_r45_direct_se.PARAM_indim:neko_get_arg(this.PARAM_indim,params,3),
                neko_assess_r45_direct_se.PARAM_outdim: neko_get_arg(this.PARAM_outdim, params, 384),
                neko_assess_r45_direct_se.PARAM_sedim: neko_get_arg(this.PARAM_sedim, params, 128),
            },MN.ASSESSMENT_FE(prfx) # although its global, but since this thing operates on a global level.
        );
        modcfg = this.config_saveable(
            modcfg,
            spatial_attention_NG_mk1,
            {"ifc": 16,
             "n_parts": 1,
             "feat_ch": 512,
             "cam_ch": 64,
             "num_se_channels": 64,
             "detached": False
             },  MN.ASSESSMENT_ATTN(prfx));

        bogocfg[MN.ASSESSMENT_AGGR(prfx)] = {
            "bogo_mod": gen4_featmap_to_feat_abstract,
            "args":
                {
                    "mod_cvt":
                        {
                            "aggr":MN.ASSESSMENT_ATTN(prfx),
                        },
                }
        }
        return modcfg,bogocfg;

    def arm_agt_core(this,mod_prfx_dict,data_prfx_dict,agtcfg,agt_prfx,params=None):
        th=neko_get_arg(this.PARAM_thumbnail_height,params,32);
        tw = neko_get_arg(this.PARAM_thumbnail_width, params, 32);
        modprfx=neko_get_arg(this.MOD_PRFX_local,params);
        dataprfx=neko_get_arg(this.DATA_PRFX_local,params);
        agtcfg=awa.append_agent_to_cfg(agtcfg,
                                AN.ASS(dataprfx),this.get_assessment(modprfx,dataprfx,th,tw))
        return agtcfg;

class neko_thumbnail_watcher_aio(virtual_factory):
    pass