# Standard Imports
from logging import fatal

from neko_2025_NGNW.common.factories.presets.core import neko_object320_factory_core
from neko_sdk.cfgtool.argsparse import neko_get_arg
# Framework / SDK Imports (neko_sdk)
from neko_sdk.neko_framework_NG.names import P

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_2025_NGNW.common.object_32x_presets.templates.task_processing_profile import neko_task_processing

from neko_2025_NGNW.common.factories.typical_parts.heads.osctc_head_icb import neko_object320_osctc_head_factory
from neko_2025_NGNW.common.factories.typical_parts.heads.oslpos_head import neko_object320_oslpos_head_factory
class neko_abstract_roiseq_factory:
    def set_head_factory(this):
        this.head = neko_object320_osctc_head_factory(this.core.platform_cfg, this.core.gmodcfg, {}); # just a place holder and type hint for abc to make ide give proper hints
        fatal("not impl");
    def __init__(this,core:neko_object320_factory_core,params):
        this.core=core;
        this.set_head_factory();
    def arm_unique_mod_for_seqcls_task(this, modcfg, bogocfg,tcfg):
        pipeline_prefix=tcfg[neko_task_processing.MOD_PRFX];

        modcfg, bogocfg =  this.head.arm_head_mods(modcfg, bogocfg, pipeline_prefix,tcfg);
        return modcfg, bogocfg;
    def arm_training_and_pred_agents(this, acfg, tcfg): # is there a better name, like defacto_meta_
        acfg_TR = awa.empty();
        # we do the wrapping here bcs there can be more than one heads
        this.head.arm_training_and_pred_agents(acfg_TR, tcfg[neko_task_processing.META_DATA_PRFX], tcfg[neko_task_processing.MOD_PRFX], tcfg[neko_task_processing.DATA_DATA_PRFX],
                                                           tcfg[neko_task_processing.INFR_EP_PRFX],tcfg[neko_task_processing.TRAINING_EP_PRFX]);

        return awa.append_agent_to_cfg(acfg,P("training_task",tcfg[neko_task_processing.NAME]),acfg_TR);
    def arm_pred_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix):
        return this.head.arm_pred_agent(acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx, endpoint_data_prefix);

# if the head impl does not need customization or reguralrization terms to be mounted along...
class neko_ctc_roiseq_factory(neko_abstract_roiseq_factory):
    def set_head_factory(this):
        this.head = neko_object320_osctc_head_factory(this.core.platform_cfg, this.core.gmodcfg, {});
class neko_ctc_roiseq_factory_sharing(neko_ctc_roiseq_factory):
    PARAM_share_mapping = "share_mapping";
    def __init__(this,core:neko_object320_factory_core,params):
        super().__init__(core,params);
        this.share_mapping=neko_get_arg(this.PARAM_share_mapping,params,{});
    def arm_unique_mod_for_seqcls_task(this, modcfg, bogocfg,tcfg):
        pipeline_prefix=tcfg[neko_task_processing.MOD_PRFX];
        if(pipeline_prefix not in this.share_mapping):
            modcfg, bogocfg =  this.head.arm_head_mods(modcfg, bogocfg, pipeline_prefix,tcfg);
        return modcfg, bogocfg;
    def arm_training_and_pred_agents(this, acfg, tcfg): # is there a better name, like defacto_meta_
        acfg_TR = awa.empty();
        mprfx=tcfg[neko_task_processing.MOD_PRFX];
        if (mprfx in this.share_mapping):
            mprfx=this.share_mapping[mprfx];
        # we do the wrapping here bcs there can be more than one heads
        this.head.arm_training_and_pred_agents(acfg_TR, tcfg[neko_task_processing.META_DATA_PRFX], mprfx, tcfg[neko_task_processing.DATA_DATA_PRFX],
                                                           tcfg[neko_task_processing.INFR_EP_PRFX],tcfg[neko_task_processing.TRAINING_EP_PRFX]);

        return awa.append_agent_to_cfg(acfg,P("training_task",tcfg[neko_task_processing.NAME]),acfg_TR);
    def arm_pred_agent(this,acfg,meta_data_prefix,endpoint_mod_prfx,feat_data_prfx,endpoint_data_prefix):
        if (endpoint_mod_prfx in this.share_mapping):
            endpoint_mod_prfx=this.share_mapping[endpoint_mod_prfx];
        return this.head.arm_pred_agent(acfg, meta_data_prefix, endpoint_mod_prfx, feat_data_prfx, endpoint_data_prefix);


class neko_lpos_roiseq_factoy(neko_abstract_roiseq_factory):
    def set_head_factory(this):
        # this head assumes a predicted length with a time stamp aligned sequence.
        this.head = neko_object320_oslpos_head_factory(this.core.platform_cfg, this.core.gmodcfg, {});