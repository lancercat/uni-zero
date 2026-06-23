from typing import Dict

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.agents.utils.neko_squeeze_agent import neko_unsqueeze_dims_agent
# Local / Project-Specific Imports (neko_2025_NGNW, osocrNG)
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN

from neko_2025_NGNW.common.object_32x_presets.templates.data_processing_profile import neko_datastream_processing_profile,neko_anchor_process_profile

from neko_2025_NGNW.common.factories.components.fefs.featsampling.item.neko_simple_att_feat_aggr import neko_simple_f2v_aagr_part_factory
from neko_2025_NGNW.common.object_32x_presets.templates.dataspec import neko_basic_im_dataspec
from neko_2025_NGNW.common.factories.presets.fe import neko_object320_fef_abstract, neko_object320_factory_core, \
    neko_object320_fef_fill

from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_object320_feitem_factory:
    def __init__(this, core:neko_object320_factory_core,params):
        this.core=core;
        this.fef_item=this.mkfef_itemim();
        this.set_feat2vec_factory();

    def set_feat2vec_factory(this):
        this.f2vf = neko_simple_f2v_aagr_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});


    def mkfef_itemim(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_fill(this.core);
        pass;
    def arm_listim_vec_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fef_item.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        modcfg,bogocfg=this.f2vf.arm_ftwr_fvec_mods(modcfg,bogocfg,pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_vec_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.fef_item.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx);
        acfg=this.f2vf.arm_ftwr_fvec_agts(acfg,data_prfx,mod_prfx);
        return acfg;
    # public facing apis
    def arm_listim_vec_data_mod(this, modcfg, bogocfg, shared_mod_prfx,dhcfg):
        assert(len(dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES])==1);
        dname=dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES][0];
        anc_cfg=dhcfg[neko_datastream_processing_profile.ANCHOR_CFG_DICT][dname];
        data_prfx=anc_cfg[neko_anchor_process_profile.DATA_PRFX];
        modprfx=anc_cfg[neko_anchor_process_profile.MOD_PRFX];
        imhw=anc_cfg[neko_anchor_process_profile.DATA_SPEC][neko_basic_im_dataspec.SIZE_hw]; # we will figure out the cfg later
        return this.arm_listim_vec_mod(modcfg, bogocfg,modprfx, shared_mod_prfx,imhw); # bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_vec_data_agt(this,acfg,dhcfg):
        dname = dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES][0];
        anc_cfg = dhcfg[neko_datastream_processing_profile.ANCHOR_CFG_DICT][dname];
        data_prfx = anc_cfg[neko_anchor_process_profile.DATA_PRFX];
        modprfx = anc_cfg[neko_anchor_process_profile.MOD_PRFX];
        acfg=this.arm_listim_vec_agt(acfg,data_prfx,modprfx);
        acfg = awa.append_agent_to_cfg(acfg, AN.C2FKSEQ(data_prfx), neko_unsqueeze_dims_agent.get_agtcfg(
            VN.I_FEAT_VEC(data_prfx), VN.ROI_FEAT_SEQ(data_prfx), 1
        )); # fake a timestamp so we can treat the char as a text line.
        return acfg;
class neko_object320_feitem_factory_sharing(neko_object320_feitem_factory):
    PARAM_fe_share_mapping="fe_share_mapping";
    PARAM_f2v_share_mapping = "f2v_share_mapping";

    def __init__(this, core:neko_object320_factory_core,params):
        super().__init__(core,params);
        this.set_sharing(params)
    def set_sharing(this,params):
        this.shared_fe_mapping :Dict[str,str]=neko_get_arg(this.PARAM_fe_share_mapping,params,{});
        this.shared_f2v_mapping :Dict[str,str]=neko_get_arg(this.PARAM_f2v_share_mapping,params,{});


    def arm_listim_vec_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        if(pipeline_mod_prefix not in this.shared_fe_mapping):
            modcfg,bogocfg=this.fef_item.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        if(pipeline_mod_prefix not in this.shared_f2v_mapping):
            modcfg,bogocfg=this.f2vf.arm_ftwr_fvec_mods(modcfg,bogocfg,pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_vec_agt(this, acfg, data_prfx, mod_prfx):
        mod_prfx_fef=this.shared_fe_mapping.get(mod_prfx,mod_prfx);
        mod_prfx_f2vf = this.shared_f2v_mapping.get(mod_prfx, mod_prfx);
        acfg = this.fef_item.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx_fef);
        acfg=this.f2vf.arm_ftwr_fvec_agts(acfg,data_prfx,mod_prfx_f2vf);
        return acfg;

