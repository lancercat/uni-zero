# Standard Imports
from logging import fatal

from neko_2025_NGNW.common.object_32x_presets.templates.data_processing_profile import neko_datastream_processing_profile,neko_anchor_process_profile
from neko_2025_NGNW.common.factories.components.fefs.featsampling.sequence.squeeze_and_scale import \
    neko_simple_f2seq_sqeeze_short_part_factory
from neko_2025_NGNW.common.object_32x_presets.templates.dataspec import neko_basic_im_dataspec
from neko_2025_NGNW.common.factories.presets.fe import neko_object320_fef_abstract, neko_object320_factory_core, \
    neko_object320_fef_cntrocr
from neko_sdk.cfgtool.argsparse import neko_get_arg


# fe seq is tied together for future installation of routing enabled factories.
# to this reason, segmentation tasks for ocr will stay heads,
# standalone segmentation tasks will come to different pipelines.

class neko_object320_feseq_factory_abstract:
    def __init__(this, core:neko_object320_factory_core,params):
        this.core=core;
        this.fef_rois=this.mkfef_roiseq();
        this.set_feat2roiseq_factory();

    def set_feat2roiseq_factory(this):
        this.f2seq = neko_simple_f2seq_sqeeze_short_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});
        # the above is just a type hint
        fatal("not impl");
        # (this.platform_cfg,this.gmodcfg,{});

    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        fatal("not impl");
        return neko_object320_fef_stn(this.core);
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        fatal("not impl");
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):
        fatal("not impl");
        return acfg;
    def arm_listim_roiseq_data_mod(this, modcfg, bogocfg, shared_mod_prfx,dhcfg):
        assert(len(dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES])==1);
        dname=dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES][0];
        anc_cfg=dhcfg[neko_datastream_processing_profile.ANCHOR_CFG_DICT][dname];
        data_prfx=anc_cfg[neko_anchor_process_profile.DATA_PRFX];
        modprfx=anc_cfg[neko_anchor_process_profile.MOD_PRFX];
        imhw=anc_cfg[neko_anchor_process_profile.DATA_SPEC][neko_basic_im_dataspec.SIZE_hw]; # we will figure out the cfg later
        return this.arm_listim_roiseq_mod(modcfg, bogocfg,modprfx, shared_mod_prfx,imhw); # bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_roiseq_data_agt(this,acfg,dhcfg):
        dname = dhcfg[neko_datastream_processing_profile.ANCHOR_NAMES][0];
        anc_cfg = dhcfg[neko_datastream_processing_profile.ANCHOR_CFG_DICT][dname];
        data_prfx = anc_cfg[neko_anchor_process_profile.DATA_PRFX];
        modprfx = anc_cfg[neko_anchor_process_profile.MOD_PRFX];
        acfg=this.arm_listim_roiseq_agt(acfg,data_prfx,modprfx);
        return acfg;

class neko_object320_feseq_factory_one_f2seq(neko_object320_feseq_factory_abstract):
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        modcfg,bogocfg=this.f2seq.arm_ftwr_fseq_mods(modcfg,bogocfg,pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx);
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,mod_prfx);
        return acfg;

class neko_object320_feseq_factory_stn_sqz(neko_object320_feseq_factory_one_f2seq):
    def set_feat2roiseq_factory(this):
        this.f2seq = neko_simple_f2seq_sqeeze_short_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});
        # (this.platform_cfg,this.gmodcfg,{});
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_stn(this.core);

class neko_object320_feseq_factory_stn_nr_sqz(neko_object320_feseq_factory_one_f2seq):
    def set_feat2roiseq_factory(this):
        this.f2seq = neko_simple_f2seq_sqeeze_short_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});
        # (this.platform_cfg,this.gmodcfg,{});
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_stn_nr(this.core);

class neko_object320_feseq_factory_cntr_sqz(neko_object320_feseq_factory_one_f2seq):
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_cntrocr(this.core);
    def set_feat2roiseq_factory(this):
        this.f2seq = neko_simple_f2seq_sqeeze_short_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});
        # (this.platform_cfg,this.gmodcfg,{});


from neko_2025_NGNW.common.factories.components.fefs.tfe.detached_danish import neko_danish_tfe_mod_factory,neko_danish_tfe_agt_factory,neko_danish_tfe_GT_agt_factory
from neko_2025_NGNW.common.factories.components.fefs.featsampling.sequence.cam import neko_simple_f2seq_attn_part_factory
# with tfe in the wna flavor now.
class neko_object320_feseq_wnaish_factory_cntr_lcam(neko_object320_feseq_factory_one_f2seq):
    def __init__(this, core:neko_object320_factory_core,params):
        super().__init__(core, params);
        this.set_tfe_factories();


    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_cntrocr(this.core);
    def set_tfe_factories(this):
        this.tfe_modf=neko_danish_tfe_mod_factory(this.core.gmodcfg,{});
        this.tfe_agtf=neko_danish_tfe_agt_factory();

    def set_feat2roiseq_factory(this):
        this.f2seq = neko_simple_f2seq_attn_part_factory(this.core.platform_cfg, this.core.gmodcfg, {});
        # (this.platform_cfg,this.gmodcfg,{});
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        modcfg,bogocfg=this.f2seq.arm_ftwr_fseq_mods(modcfg,bogocfg,pipeline_mod_prefix);
        modcfg, bogocfg = this.tfe_modf.arm_module({
            this.tfe_modf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg, {});
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.


    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):
        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx);
        acfg=this.tfe_agtf.arm_agt_core(
            {this.tfe_modf.MOD_PRFX_local:mod_prfx},
            {this.tfe_agtf.DATA_PRFX_feat:data_prfx,
             this.tfe_agtf.DATA_PRFX_local:data_prfx},acfg,data_prfx,{});
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,data_prfx,mod_prfx);
        return acfg;



class neko_object320_feseq_wnaish_factory_cntr_lcamXA(neko_object320_feseq_wnaish_factory_cntr_lcam):
    def set_tfe_factories(this):
        this.tfe_modf=neko_danish_tfe_2XL_mod_factory(this.core.gmodcfg,{});
        this.tfe_agtf=neko_danish_tfe_agt_factory();
    # XA has a large att map.
    pass;


# a hack forcing lsct to share tfe with main branch
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack(neko_object320_feseq_wnaish_factory_cntr_lcam):
    PARAM_share_mapping="share_mapping";
    def set_share(this,params):
        this.share_mapping = neko_get_arg(this.PARAM_share_mapping, params, {
            "lsct_ostr": "ostrv1_data"
        });
    def __init__(this, core:neko_object320_factory_core,params):
        super().__init__(core, params);
        this.set_share(params);
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        modcfg,bogocfg=this.f2seq.arm_ftwr_fseq_mods(modcfg,bogocfg,pipeline_mod_prefix);
        if (pipeline_mod_prefix not in this.share_mapping):
            modcfg, bogocfg = this.tfe_modf.arm_module({
                this.tfe_modf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg, {});
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):

        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx);
        if (mod_prfx in this.share_mapping):
            tfe_mod_prfx=this.share_mapping[mod_prfx];
        else:
            tfe_mod_prfx=mod_prfx;
        acfg=this.tfe_agtf.arm_agt_core(
            {this.tfe_modf.MOD_PRFX_local:tfe_mod_prfx},
            {this.tfe_agtf.DATA_PRFX_feat:data_prfx,
             this.tfe_agtf.DATA_PRFX_local:data_prfx},acfg,data_prfx,{});
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,data_prfx,mod_prfx);
        return acfg;





class neko_object320_feseq_wnaish_factory_cntr_lcam_GT_lsct_share_tfe_hack(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack):
    def set_tfe_factories(this):
        this.tfe_modf=neko_danish_tfe_mod_factory(this.core.gmodcfg,{});
        this.tfe_agtf=neko_danish_tfe_GT_agt_factory(); # temoral gradient attached
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack):
    def set_share(this,params):
        this.share_mapping = neko_get_arg(this.PARAM_share_mapping, params, {
            "lsct_ostr": "ostrv1_data",
            "notbg_lsct": "istr_notbg"
        });
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_attn_indic_abl(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack):
    def set_share(this,params):
        this.share_mapping = neko_get_arg(this.PARAM_share_mapping, params, {
            "lsct_ostr": "ostrv1_data",
            "notbg_lsct": "istr_notbg",
            "istr_punjabi": "istr_notbg",
            "istr_tamil": "istr_notbg",
        });

## this also forces a head sharing
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_attn_full_indic_abl(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_attn_indic_abl):
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        if (pipeline_mod_prefix not in this.share_mapping):
            modcfg, bogocfg = this.tfe_modf.arm_module({
                this.tfe_modf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg, {});
            modcfg, bogocfg = this.f2seq.arm_ftwr_fseq_mods(modcfg, bogocfg, pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.

    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):

        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,mod_prfx);
        if (mod_prfx in this.share_mapping):
            tfe_mod_prfx=this.share_mapping[mod_prfx];
        else:
            tfe_mod_prfx=mod_prfx;
        acfg=this.tfe_agtf.arm_agt_core(
            {this.tfe_modf.MOD_PRFX_local:tfe_mod_prfx},
            {this.tfe_agtf.DATA_PRFX_feat:data_prfx,
             this.tfe_agtf.DATA_PRFX_local:data_prfx},acfg,data_prfx,{});
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,data_prfx,tfe_mod_prfx);
        return acfg;

class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_bn_indic_abl(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack):
    PARAM_shared_bn_mapping="shared_bn_mapping";
    def __init__(this, core:neko_object320_factory_core,params):
        super().__init__(core, params);
        this.set_shared_bn(params);
    def set_share(this,params):
        this.share_mapping = neko_get_arg(this.PARAM_share_mapping, params, {
            "lsct_ostr": "ostrv1_data",
            "notbg_lsct": "istr_notbg",
            "istr_punjabi": "istr_notbg",
            "istr_tamil": "istr_notbg",
        });

    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        if (pipeline_mod_prefix not in this.shared_bn_mapping):
            modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        if (pipeline_mod_prefix not in this.share_mapping):
            modcfg, bogocfg = this.tfe_modf.arm_module({
                this.tfe_modf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg, {});
        modcfg,bogocfg=this.f2seq.arm_ftwr_fseq_mods(modcfg,bogocfg,pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.
    def set_shared_bn(this,params):
        this.shared_bn_mapping = neko_get_arg(this.PARAM_shared_bn_mapping, params, {
            "istr_punjabi": "istr_notbg",
            "istr_tamil": "istr_notbg",
        });
    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):
        if (mod_prfx in this.shared_bn_mapping):
            ife_mod_prfx=this.shared_bn_mapping[mod_prfx];
        else:
            ife_mod_prfx =mod_prfx;
        if (mod_prfx in this.share_mapping):
            tfe_mod_prfx=this.share_mapping[mod_prfx];
        else:
            tfe_mod_prfx=mod_prfx;
        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,ife_mod_prfx);
        acfg=this.tfe_agtf.arm_agt_core(
            {this.tfe_modf.MOD_PRFX_local:tfe_mod_prfx},
            {this.tfe_agtf.DATA_PRFX_feat:data_prfx,
             this.tfe_agtf.DATA_PRFX_local:data_prfx},acfg,data_prfx,{});
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,data_prfx,mod_prfx);
        return acfg;
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_all_indic_abl(neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_share_bn_indic_abl):
    PARAM_shared_attn_head_mapping="shared_attn_head_mapping";
    def __init__(this, core:neko_object320_factory_core,params):
        super().__init__(core, params);
        this.set_shared_attn_head(params)
    def arm_listim_roiseq_mod(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx,imhw):
        if (pipeline_mod_prefix not in this.shared_bn_mapping):
            modcfg,bogocfg=this.fef_rois.arm_listim_ftwr_mod(modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx, imhw);
        if (pipeline_mod_prefix not in this.share_mapping):
            modcfg, bogocfg = this.tfe_modf.arm_module({
                this.tfe_modf.MOD_PRFX_local: pipeline_mod_prefix}, modcfg, bogocfg, {});
        if (pipeline_mod_prefix not in this.shared_attn_head_mapping):
            modcfg,bogocfg=this.f2seq.arm_ftwr_fseq_mods(modcfg,bogocfg,pipeline_mod_prefix);
        return modcfg,bogocfg# bcs here collate does not add new models, in tps based collate, it will.
    def set_shared_attn_head(this,params):
        this.shared_attn_head_mapping = neko_get_arg(this.PARAM_shared_attn_head_mapping, params, {
            "istr_punjabi": "istr_notbg",
            "istr_tamil": "istr_notbg",
        });
    def arm_listim_roiseq_agt(this, acfg, data_prfx, mod_prfx):
        if (mod_prfx in this.shared_bn_mapping):
            ife_mod_prfx=this.shared_bn_mapping[mod_prfx];
        else:
            ife_mod_prfx =mod_prfx;
        if (mod_prfx in this.share_mapping):
            tfe_mod_prfx=this.share_mapping[mod_prfx];
        else:
            tfe_mod_prfx=mod_prfx;
        if(mod_prfx in this.shared_attn_head_mapping):
            attn_mod_prfx=this.shared_attn_head_mapping[mod_prfx];
        else:
            attn_mod_prfx=mod_prfx;
        acfg = this.fef_rois.arm_listim_ftwr_agt(acfg,data_prfx,ife_mod_prfx);
        acfg=this.tfe_agtf.arm_agt_core(
            {this.tfe_modf.MOD_PRFX_local:tfe_mod_prfx},
            {this.tfe_agtf.DATA_PRFX_feat:data_prfx,
             this.tfe_agtf.DATA_PRFX_local:data_prfx},acfg,data_prfx,{});
        acfg=this.f2seq.arm_ftwr_fseq_agts(acfg,data_prfx,data_prfx,attn_mod_prfx);
        return acfg;


class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind_stn(
    neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind):
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_stn(this.core);
class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind_stnE(
    neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind):
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_stnE(this.core);

class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind_stnECI(
    neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind):
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_stnECI(this.core);

class neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind_tps(
    neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind):
    def mkfef_roiseq(this) -> neko_object320_fef_abstract:
        return neko_object320_fef_tps(this.core);

