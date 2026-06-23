# seq flatten is done with prediction and head where they take in length.
# its not needed in non-terminal states either way.
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,global_mod_cfg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory
from neko_sdk.OSR.classification.thresholder import neko_tanh_thresholder_static
from neko_sdk.OSR.classification.agents.unk_cat import neko_inject_cos_thresh_agent, neko_inject_cos_thresh_2d_agent
from neko_sdk.neko_framework_NG.agents.utils.ops import neko_list_mul_agent
from osocrNG.data_utils.common_data_presets_mk4.presets.names import P
from neko_sdk.OSR.classification.agents.case_sim_reduction import neko_proto_sim_reduction_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.libsim.modules.neko_cosconf_sim_NG import neko_cosconf_sim_NG,neko_2d_sp_cosconf_sim_NG
from neko_sdk.neko_framework_NG.libsim.agents.multi_part_cosconf import neko_multipart_sim_sample_proto

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

# let me be clear the prediction adn reward does NOT happen here, as we now have multi meta sources
class classification_factory_mk2_mod(virtual_mod_factory):
    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        mprfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_saveable(modcfg,neko_tanh_thresholder_static,{},MN.UNK_TH(mprfx));
        modcfg=this.config_saveable(modcfg,neko_cosconf_sim_NG,{},MN.SIM(mprfx));
        return modcfg,bogocfg;



# maybe in 32x we can remove the l part to head?
class classification_factory_mk2_agt(virtual_agt_factory):
    DATA_PRFX_feat="feat";
    DATA_PRFX_meta="meta";
    def logit_dim(this):
        return -1;

    def get_src_name(this,prfx):
        return VN.ROI_FEAT_SEQ(prfx);
    def arm_unk_agt(this, lacfg, logit_prfx, mprfx):
        return awa.append_agent_to_cfg(lacfg,"arm_unk",neko_inject_cos_thresh_agent.get_agtcfg(
            VN.COSSIM(logit_prfx),VN.COSSIM_WUNK(logit_prfx),MN.UNK_TH(mprfx)
        ));
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        feat_prfx=data_prfx_dict[this.DATA_PRFX_feat];
        meta_prfx=data_prfx_dict[this.DATA_PRFX_meta];
        logit_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict,P)
        mprfx=mod_prfx_dict[this.MOD_PRFX_local];
        lacfg = awa.empty([VN.SAM_ID(feat_prfx),VN.PROTO(meta_prfx)]);
        lacfg=awa.append_agent_to_cfg(lacfg,"sim",neko_multipart_sim_sample_proto.get_agtcfg(
            this.get_src_name(feat_prfx),VN.PROTO(meta_prfx),
            VN.COSSIM(logit_prfx),VN.SIMCONF(logit_prfx),
            MN.SIM(mprfx)
        ));
        lacfg=this.arm_unk_agt(lacfg, logit_prfx, mprfx);
        lacfg=awa.append_agent_to_cfg(lacfg,"mklogit",neko_list_mul_agent.get_agtcfg(
            [VN.COSSIM_WUNK(logit_prfx),VN.SIMCONF(logit_prfx)],VN.DENSE_CENTER_PRED_LOGIT(logit_prfx)
        ));
        lacfg=awa.append_agent_to_cfg(lacfg,"mkclslogit",neko_proto_sim_reduction_agent.get_agtcfg(
            VN.PROTO_LABEL(meta_prfx),VN.DENSE_CENTER_PRED_LOGIT(logit_prfx),VN.DENSE_CLS_PRED_LOGIT(logit_prfx),this.logit_dim()
        ));
        agtcfg=awa.append_agent_to_cfg(agtcfg,AN.PRED(agt_prfx),lacfg)
        return agtcfg

class classification_factory_mk2_factory(virtual_part_factory):
    def setup_factories(this):
        this.classification_mod=classification_factory_mk2_mod(this.gmodcfg,{});
        this.classification_agt=classification_factory_mk2_agt();



