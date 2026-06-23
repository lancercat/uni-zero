import os
from logging import fatal

# Framework / SDK Imports
from neko_sdk.neko_framework_NG.names import P
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_sdk.neko_framework_NG.agents.neko_loss_backward_all_agent import get_neko_basic_backward_all_agent
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import (
    neko_agent_wrapping_agent as awa,
    neko_agent_wrapping_agent_nograd as awa_nograd
)

# Project-Specific Imports: Templates & Configs
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.templates.meta_processing_profile import neko_metastream_processing
from neko_2025_NGNW.common.object_32x_presets.templates.task_processing_profile import neko_task_processing
from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.data_processing_profile import (
    neko_datastream_processing_profile,
    neko_anchor_process_profile
)
from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile
from neko_2025_NGNW.common.object_32x_presets.templates.tome import (
    neko_320_tome,
    neko_320_fpbpgrp_tome,
    neko_simple_320_test_task_tome
)

# Project-Specific Imports: Factories
from neko_2025_NGNW.common.factories.presets.core import neko_object320_factory_core, neko_object320_factory_core_ffn, \
    neko_object320_factory_core_nf
# from neko_2025_NGNW.common.factories.presets.semseg_head import neko_dense_val_semseg_factory, \
#     neko_dese_val_semseg_factor_corr05, neko_dese_val_semseg_bga_factor_corr05
# from neko_2025_NGNW.common.factories.presets.ac_insseg_head import neko_first_gtlen_aux_cls_agnostic_insseg_factory
# from neko_2025_NGNW.common.factories.typical_parts.heads.osic_head_icb import neko_object320_osic_head_factory

from neko_2025_NGNW.common.factories.typical_parts.heads.osic_head_icb import neko_object320_osic_head_factory

from neko_2025_NGNW.common.factories.typical_parts.cross_task_training.xtcs import neko_cross_task_cos_sim_factory
from neko_2025_NGNW.common.factories.components.debug.result_vis_factory import debug_vis_agent_factory
from neko_2025_NGNW.common.factories.components.prototyper.as_proto import neko_as_prototype_factory
# from neko_2025_NGNW.common.factories.components.prototyper.id_tied_centers import neko_embedding_pool_aggr_part_factory
# from neko_2025_NGNW.common.factories.presets.seqcls_head import neko_lpos_roiseq_factoy, neko_ctc_roiseq_factory
from neko_2025_NGNW.common.factories.presets.im2vec import neko_object320_feitem_factory


# Project-Specific Imports: Agents
from neko_2025_NGNW.common.agents.training_and_testing.neko_testing_core_agent import neko_task_testing_core_agent
from neko_2025_NGNW.common.agents.training_and_testing.neko_32x_meta_caching_agent import neko_320_im_meta_caching_agent
from osocrNG.modular_agents_ocrNG.pred_subs.gt_unk_making_agents import translate_gt_tokens_list_agent
from osocrNG.modular_agents_ocrNG.logging_mk2.acr_agent_mk2 import acr_fps_reporter_mk2
from osocrNG.modular_agents_ocrNG.debugging_agents.log_saver.simple_sample_image_saver import (
    neko_simple_img_logging_agent,
    neko_simple_seq_logging_agent
)
from osocrNG.modular_agents_ocrNG.debugging_agents.log_saver.simple_sample_tensor_image_saver import neko_simple_tensor_img_logging_agent

# we need a way for routed fef.
class neko_object320_oscr_no_route_factory:

    def set_protofy_factory(this):
        this.protofy_factory=neko_as_prototype_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    # def set_embedding_factory(this):
    #     this.embedding_factory=neko_embedding_pool_aggr_part_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    def set_imcls_head_factory(this):
        this.imcls_head=neko_object320_osic_head_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    def set_xcts_factory(this):
        this.xcts_factory=neko_cross_task_cos_sim_factory(this.core.platform_cfg,this.core.gmodcfg,{});

    def set_debugging_factory(this):
        this.dbgagtf=debug_vis_agent_factory();
    def mk_core(this, platform_cfg: neko_platform_cfg, params)->neko_object320_factory_core:
        return neko_object320_factory_core_ffn(platform_cfg, params);
    # templates are no longer registered here.

    def mkfeseq_roiseq(this) :
        fatal("not impl")
    def mkfeseq_item(this)->neko_object320_feitem_factory:
        return neko_object320_feitem_factory(this.core,{});
    def mkfeseqcls_head(this):
        return None
    def mksemseg_head(this):
        return neko_dense_val_semseg_factory(this.core,{});
    def mkca_insseg_head(this):
        return neko_first_gtlen_aux_cls_agnostic_insseg_factory(this.core,{});

    def __init__(this, platform_cfg: neko_platform_cfg, params):
        this.core=this.mk_core(platform_cfg, params);
        # this.feseqf=this.mkfeseq_roiseq();
        this.feitemf=this.mkfeseq_item();
        # this.fetfef=this.mkfetfe(); # this is to just make fe and tfe for sem/ins-seg.
        # this.seqcls_head=this.mkfeseqcls_head();
        # this.semseg_head=this.mksemseg_head(); # instance segmentation will have a different head, there will be a third head to combine both.
        # this.ordered_ca_insseg_head=this.mkca_insseg_head();

        # core stuff
        this.set_imcls_head_factory();
        this.set_xcts_factory();
        this.set_debugging_factory();
        this.set_protofy_factory();
        # this.set_embedding_factory();

    def arm_data_mod_anchor(this):
        pass;

    # public facing apis
    def arm_word_data_mod(this, modcfg, bogocfg, shared_mod_prfx,dhcfg):
        pass;



    def arm_listim_proto_mod(this, modcfg, bogocfg,  shared_mod_prfx,mcfg):
        mod_prfx = mcfg[neko_metastream_processing.MOD_PRFX];
        if(not mcfg[neko_metastream_processing.FE_MOD_TIED]):
            modcfg,bogocfg=this.feitemf.arm_listim_vec_mod(modcfg, bogocfg, mod_prfx, shared_mod_prfx,mcfg[neko_metastream_processing.SIZE]);
        modcfg,bogocfg=this.protofy_factory.arm_mod_pipeline(modcfg,bogocfg,mod_prfx);
        return modcfg,bogocfg;
    def arm_id_proto_mod(this, modcfg, bogocfg,  shared_mod_prfx,mcfg):
        mod_prfx = mcfg[neko_metastream_processing.MOD_PRFX];
        modcfg, bogocfg = this.embedding_factory.id_emb_modf.arm_module(
            {this.embedding_factory.MOD_PRFX_local:mod_prfx},modcfg, bogocfg,{});
        modcfg,bogocfg=this.protofy_factory.arm_mod_pipeline(modcfg,bogocfg,mod_prfx);
        return modcfg, bogocfg;


    def arm_id_proto_agt(this, acfg,mcfg):
        data_prfx=mcfg[neko_metastream_processing.DATA_PRFX];
        mod_prfx = mcfg[neko_metastream_processing.MOD_PRFX];
        assert (len(mcfg[neko_metastream_processing.MODSCP_ENDPOINTS])==1);
        armed_proto_data_prfx=mcfg[neko_metastream_processing.MODSCP_ENDPOINTS][0]
        acfg=this.embedding_factory.arm_id_vec_agt(
            acfg, data_prfx, mod_prfx
        )
        acfg=this.protofy_factory.arm_proto_arming_agent(acfg,data_prfx,armed_proto_data_prfx,mod_prfx);
        return acfg;

    def arm_listim_proto_agt(this, acfg,mcfg):
        data_prfx=mcfg[neko_metastream_processing.DATA_PRFX];
        mod_prfx = mcfg[neko_metastream_processing.MOD_PRFX];
        assert (len(mcfg[neko_metastream_processing.MODSCP_ENDPOINTS])==1);
        armed_proto_data_prfx=mcfg[neko_metastream_processing.MODSCP_ENDPOINTS][0]
        acfg=this.feitemf.arm_listim_vec_agt(acfg,data_prfx,mod_prfx);
        acfg=this.protofy_factory.arm_proto_arming_agent(acfg,data_prfx,armed_proto_data_prfx,mod_prfx);
        return acfg;


    def arm_unique_mod_for_imcls_task(this, modcfg, bogocfg,tcfg):
        pipeline_prefix=tcfg[neko_task_processing.MOD_PRFX];
        modcfg, bogocfg =  this.imcls_head.arm_head_mods(modcfg, bogocfg, pipeline_prefix,tcfg);
        return modcfg, bogocfg;

    # just one task
    # modscp = mode descriptor, which the name comes from llm
    # i want to call it defacto meta but eventually think there can be something more decent
    # https://gemini.google.com/share/105bb26c6c02
    def arm_one_oscr_task_training_agent(this, acfg, tcfg): # is there a better name, like defacto_meta_
        acfg_TR = awa.empty();
        # we do the wrapping here bcs there can be more than one heads
        this.imcls_head.arm_training_and_pred_agents(
            acfg_TR, tcfg[neko_task_processing.META_DATA_PRFX], tcfg[neko_task_processing.MOD_PRFX],
            tcfg[neko_task_processing.DATA_DATA_PRFX],
            tcfg[neko_task_processing.INFR_EP_PRFX],tcfg[neko_task_processing.TRAINING_EP_PRFX],);

        return awa.append_agent_to_cfg(acfg,P("training_task",tcfg[neko_task_processing.NAME]),acfg_TR);




    def arm_taskgrp_mods(this, modcfg, bogocfg,tome:neko_320_tome):
        for dpipe in tome.data_item_im_stream_handler_profiles:
            # normally, one data ep has its own modprfx, but if the use is sure enough two sources are close enough this can be shared.
            shared_mod_prefx = this.core.global_mod_prefix;
            # there will be more complex logic once we move to the hybrid of word and char, especially after we have routing back
            modcfg, bogocfg = this.feitemf.arm_listim_vec_data_mod(modcfg, bogocfg,shared_mod_prefx,tome.data_item_im_stream_handler_profiles[dpipe]);
        for dpipe in tome.data_seq_im_stream_handler_profiles:
            # normally, one data ep has its own modprfx, but if the use is sure enough two sources are close enough this can be shared.
            shared_mod_prefx = this.core.global_mod_prefix;
            modcfg, bogocfg = this.feseqf.arm_listim_roiseq_data_mod(modcfg, bogocfg,shared_mod_prefx,tome.data_seq_im_stream_handler_profiles[dpipe]);
        # for dpipe in tome.data_panoptic_im_stream_handler_profiles:
        #     # normally, one data ep has its own modprfx, but if the use is sure enough two sources are close enough this can be shared.
        #     shared_mod_prefx = this.core.global_mod_prefix;
        #     modcfg, bogocfg = this.feseqf.arm_listim_roiseq_data_mod(modcfg, bogocfg, shared_mod_prefx,
        #                                                              tome.data_seq_im_stream_handler_profiles[
        #                                                                  dpipe]);

            # there will be more complex logic once we move to the hybrid of word and char, especially after we have routing back
            # modcfg, bogocfg = this.arm_word_data_mod(modcfg, bogocfg,shared_mod_prefx,tome.data_item_im_stream_handler_profiles[dpipe]);

        # there will be more if we introduce more modality sinfo, but we bear with this now
        for mpipe in tome.im_meta_stream_handler_profiles:
            shared_mod_prefx = this.core.global_mod_prefix;
            modcfg, bogocfg = this.arm_listim_proto_mod(modcfg, bogocfg,shared_mod_prefx,tome.im_meta_stream_handler_profiles[mpipe]);
        for mpipe in tome.longid_meta_stream_handler_profiles:
            shared_mod_prefx = this.core.global_mod_prefix;
            modcfg, bogocfg = this.arm_id_proto_mod(modcfg, bogocfg, shared_mod_prefx,
                                                        tome.longid_meta_stream_handler_profiles[mpipe]);

        for task in tome.imcls_task_handlers:
            modcfg, bogocfg = this.arm_unique_mod_for_imcls_task(modcfg, bogocfg, tome.imcls_task_handlers[task]);
        # for task in tome.ordered_roiseq_task_handlers:
        #     if(this.seqcls_head is None):
        #         print("skipping cls, this is likly caused by a debugging panoptic seg branch");
        #     else:
        #         modcfg, bogocfg = this.seqcls_head.arm_unique_mod_for_seqcls_task(modcfg, bogocfg, tome.ordered_roiseq_task_handlers[task]);
        # for task in tome.semseg_task_handlers:
        #     modcfg,bogocfg=this.semseg_head.arm_unique_mod_for_semseg_task(modcfg, bogocfg, tome.semseg_task_handlers[task]);
        # for task in tome.insseg_task_handlers:
        #     modcfg, bogocfg = this.ordered_ca_insseg_head.arm_unique_mod_for_cains_task(modcfg, bogocfg,
        #                                                                       tome.insseg_task_handlers[task]);

        return modcfg, bogocfg;

    # you will need to declare lsct in taskdscp
    def get_training_task_fpbp_grp(this, tome:neko_320_fpbpgrp_tome):
        acfg = awa.empty();
        for mname in tome.im_meta_stream_handler_profiles:
            macfg=awa.empty();
            mcfg=tome.im_meta_stream_handler_profiles[mname];
            meta_data_prfx=mcfg[neko_metastream_processing.DATA_PRFX];
            macfg = this.arm_listim_proto_agt(macfg,mcfg);
            acfg=awa.append_agent_to_cfg(acfg,meta_data_prfx,macfg);
        for mname in tome.longid_meta_stream_handler_profiles:
            macfg=awa.empty();
            mcfg=tome.longid_meta_stream_handler_profiles[mname];
            meta_data_prfx=mcfg[neko_metastream_processing.DATA_PRFX];
            macfg = this.arm_id_proto_agt(macfg,mcfg);
            acfg=awa.append_agent_to_cfg(acfg,meta_data_prfx,macfg);
        for dname in tome.data_item_im_stream_handler_profiles:
            dacfg=awa.empty();
            dprofile=tome.data_item_im_stream_handler_profiles[dname]
            dacfg=this.feitemf.arm_listim_vec_data_agt(dacfg,dprofile);
            acfg=awa.append_agent_to_cfg(acfg,dname,dacfg);
        for dname in tome.data_seq_im_stream_handler_profiles:
            dacfg = awa.empty();
            dprofile = tome.data_seq_im_stream_handler_profiles[dname]
            dacfg = this.feseqf.arm_listim_roiseq_data_agt(dacfg,dprofile);
            acfg = awa.append_agent_to_cfg(acfg, dname, dacfg);
        tacfg=awa.empty();
        for task_name in tome.imcls_task_handlers:
            tacfg=this.arm_one_oscr_task_training_agent(tacfg, tome.imcls_task_handlers[task_name]);
        # for task_name in tome.xtcs_task_handlers:
        #     tacfg=this.xcts_factory.arm_training_agent(tacfg,tome.xtcs_task_handlers[task_name])
        # for task_name in tome.ordered_roiseq_task_handlers:
        #     # note it needs data_seq_im_stream_handler_profiles head to provide tfe
        #     # you cannot apply this to item stream
        #     if(this.seqcls_head is None):
        #         print("skipping cls, this is likly caused by a debugging panoptic seg branch");
        #     else:
        #         tacfg=this.seqcls_head.arm_training_and_pred_agents(tacfg,tome.ordered_roiseq_task_handlers[task_name]);
        # for task_name in tome.semseg_task_handlers:
        #     tacfg=this.semseg_head.arm_training_and_pred_agents(tacfg,tome.semseg_task_handlers[task_name]);
        # for task_name in tome.insseg_task_handlers:
        #     # note it needs data_seq_im_stream_handler_profiles head to provide tfe
        #     # you cannot apply this to item stream
        #     tacfg = this.ordered_ca_insseg_head.arm_training_and_pred_agents(tacfg, tome.insseg_task_handlers[task_name]);

        acfg=awa.append_agent_to_cfg(acfg,"tasks",tacfg);
        acfg = awa.append_agent_to_cfg(acfg, "bp", get_neko_basic_backward_all_agent());
        return acfg

    def get_meta_cacher(this,tetome:neko_simple_320_test_task_tome,task_data_prfx,armed_prfx):
        tat = awa.empty();

        meta_loader_agent = tetome.meta_ldr_acfg;

        side_info_loader_agent = tetome.sinfo_ldr_acfg;

        meta_prfx = tetome.meta_ldr_profile[task_data_prfx][neko_meta_stream_profile.META_ENDPOINTS][0];
        sinfo_prfx = tetome.meta_ldr_profile[task_data_prfx][neko_meta_stream_profile.ENDPOINTS][0];
        sinfo_mod_prfx = tetome.meta_ldr_profile[task_data_prfx][neko_meta_stream_profile.SIDE_INFO_MBND][sinfo_prfx];

        sinfo_encoder_acfg = this.feitemf.arm_listim_vec_agt(awa.empty(), sinfo_prfx, sinfo_mod_prfx);
        post_processor_agent_cfg = this.protofy_factory.arm_proto_arming_agent(awa.empty(), sinfo_prfx, armed_prfx,
                                                                               sinfo_mod_prfx)

        return awa.append_agent_to_cfg(tat, "caching", neko_320_im_meta_caching_agent.get_agtcfg(
            VN.PROTO_LABEL(armed_prfx), VN.PROTO(armed_prfx), VN.TDICT(armed_prfx), VN.UTF(armed_prfx),
            VN.TDICT_HASH(armed_prfx), VN.META_TDICT_HASH(armed_prfx), 256, meta_loader_agent, side_info_loader_agent,
            sinfo_encoder_acfg,
            post_processor_agent_cfg,
            VN.UTF(meta_prfx), VN.TDICT(meta_prfx), VN.UTF(sinfo_prfx), VN.TDICT(sinfo_prfx),
            VN.I_FEAT_VEC(sinfo_prfx), VN.UTF(sinfo_prfx), VN.UTF(sinfo_prfx), VN.I_FEAT_VEC(sinfo_prfx)
        ));
    def get_testing_task(this, tome:neko_320_tome,logging_path=None):

        atea=awa_nograd.empty();
        # during testing, each task will have their own metas and sinfos built.
        for task_data_prfx in tome.imcls_testing_tasks:
            tetome=tome.imcls_testing_tasks[task_data_prfx];
            armed_prfx = tetome.meta_process_profile[neko_metastream_processing.MODSCP_ENDPOINTS][0];
            tat=this.get_meta_cacher(tetome,task_data_prfx,armed_prfx);
            data_loader_agent = tetome.disk_ldr_acfg;
            data_disldr = tetome.data_ldr_profile[task_data_prfx];
            data_prfx = data_disldr[neko_datastream_profile.ENDPOINTS][0];


            roi_prfx = tetome.data_process_profile[neko_datastream_processing_profile.ROI_EPS][0];
            infer_prfx = P("inference", task_data_prfx);
            log_ep_prefix = P("logging", task_data_prfx);

            net_agt=awa.empty();
            net_agt=this.feitemf.arm_listim_vec_data_agt(net_agt,tetome.data_process_profile)
            # net_agt = this.arm_listim_vec_agt(net_agt, data_prfx, tetome.data_mod_name);
            net_agt = this.imcls_head.arm_pred_agent(net_agt, armed_prfx, tetome.task_mod_name,roi_prfx,infer_prfx);
            lgr_agt=awa.wrap_this(acr_fps_reporter_mk2.get_agtcfg(
                VN.PRED_TOK(infer_prfx),VN.GT_TOK_UTF(data_prfx),VN.TDICT(armed_prfx),VN.TASKPERF(task_data_prfx),task_data_prfx,tetome.task_disable_unk));
            dbg_agt=awa.empty();
            if (logging_path):
                # dbg_agt = awa.append_agent_to_cfg(dbg_agt, "img_logger",
                #                                   neko_simple_img_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                #                                                                            [VN.IM_raw(data_prfx)],
                #                                                                            data_prfx, logging_path,
                #                                                                            ["id"])
                #                                   );
                # dbg_agt = awa.append_agent_to_cfg(dbg_agt, "col_logger",
                #                                   neko_simple_tensor_img_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                #                                                                            [VN.IM_normed_tensor(data_prfx)],
                #                                                                            data_prfx, logging_path,
                #                                                                            ["id"])
                #                                   );

                dbg_agt = awa.append_agent_to_cfg(dbg_agt, "pred_logger",
                                                  neko_simple_seq_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                                                                                           VN.PRED_TOK(infer_prfx),
                                                                                           "pred", data_prfx,
                                                                                           logging_path, ["id"])
                                                  );
                dbg_agt=awa.append_agent_to_cfg(dbg_agt,"gt_mkunk",translate_gt_tokens_list_agent.get_agtcfg(
                    VN.GT_TOK_UTF(data_prfx), VN.TDICT(armed_prfx),VN.GT_TOK_UTF_WUNK(infer_prfx)
                ));
                dbg_agt = awa.append_agent_to_cfg(dbg_agt, "gt_logger",
                                                  neko_simple_seq_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                                                                                           VN.GT_TOK_UTF_WUNK(infer_prfx),
                                                                                           "gt", data_prfx,
                                                                                           logging_path, ["id"])
                                                  );
                awa.append_agent_to_cfg(net_agt, "debug", dbg_agt);

            acfg = neko_task_testing_core_agent.get_agtcfg(data_loader_agent,net_agt,lgr_agt,task_data_prfx);



            awa.append_agent_to_cfg(tat,"payload",acfg);
            awa.append_agent_to_cfg(atea,task_data_prfx,tat);

        for task_data_prfx in tome.im_seqcls_testing_tasks:
            tetome=tome.im_seqcls_testing_tasks[task_data_prfx];
            armed_prfx = tetome.meta_process_profile[neko_metastream_processing.MODSCP_ENDPOINTS][0];
            tat=this.get_meta_cacher(tetome,task_data_prfx,armed_prfx);

            data_loader_agent = tetome.disk_ldr_acfg;
            data_disldr = tetome.data_ldr_profile[task_data_prfx];
            data_prfx = data_disldr[neko_datastream_profile.ENDPOINTS][0];


            roi_prfx = tetome.data_process_profile[neko_datastream_processing_profile.ROI_EPS][0];
            infer_prfx = P("inference", task_data_prfx);
            log_ep_prefix = P("logging", task_data_prfx);


            net_agt = awa.empty();
            net_agt = this.feseqf.arm_listim_roiseq_data_agt(net_agt, tetome.data_process_profile)
            # net_agt = this.arm_listim_vec_agt(net_agt, data_prfx, tetome.data_mod_name);
            net_agt = this.seqcls_head.arm_pred_agent(net_agt, armed_prfx, tetome.task_mod_name, roi_prfx, infer_prfx);
            lgr_agt = awa.wrap_this(acr_fps_reporter_mk2.get_agtcfg(
                VN.PRED_TOK(infer_prfx), VN.GT_TOK_UTF(data_prfx), VN.TDICT(armed_prfx), VN.TASKPERF(task_data_prfx),
                task_data_prfx,tetome.task_disable_unk));
            dbg_agt=awa.empty();

            if(logging_path):
                # dbg_agt=awa.append_agent_to_cfg(dbg_agt,"img_logger",
                #     neko_simple_img_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),[VN.IM_raw(data_prfx)],data_prfx,logging_path,["id"])
                # );
                # dbg_agt = awa.append_agent_to_cfg(dbg_agt, "col_logger",
                #                                   neko_simple_tensor_img_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                #                                                                            [VN.IM_normed_tensor(data_prfx)],
                #                                                                            data_prfx, logging_path,
                #                                                                            ["id"])
                #                                   );

                dbg_agt = awa.append_agent_to_cfg(dbg_agt, "pred_logger",neko_simple_seq_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                            VN.PRED_TOK(infer_prfx),"pred", data_prfx, logging_path,["id"])
                                                  );
                dbg_agt=awa.append_agent_to_cfg(dbg_agt,"gt_mkunk",translate_gt_tokens_list_agent.get_agtcfg(
                    VN.GT_TOK_UTF(data_prfx), VN.TDICT(armed_prfx),VN.GT_TOK_UTF_WUNK(infer_prfx)
                ));
                dbg_agt = awa.append_agent_to_cfg(dbg_agt, "gt_logger",neko_simple_seq_logging_agent.get_agtcfg(VN.SAM_ORIG(data_prfx),
                            VN.GT_TOK_UTF_WUNK(infer_prfx),"gt", data_prfx, logging_path,["id"])
                                                  );
                awa.append_agent_to_cfg(net_agt,"debug",dbg_agt);

            acfg = neko_task_testing_core_agent.get_agtcfg(data_loader_agent,net_agt,lgr_agt,task_data_prfx);

            awa.append_agent_to_cfg(tat,"payload",acfg);
            awa.append_agent_to_cfg(atea,task_data_prfx,tat);

        # for task_data_prfx in tome.semseg_task_handlers:
        #     pass;


        return atea;
class neko_object320_oscr_no_route_no_stn_factory(neko_object320_oscr_no_route_factory):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_factory_cntr_sqz(this.core,{});
class neko_object320_oscr_no_route_basectc(neko_object320_oscr_no_route_factory):
    def mk_core(this, platform_cfg: neko_platform_cfg, params)->neko_object320_factory_core:
        return neko_object320_factory_core(platform_cfg, params);
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_factory_cntr_sqz(this.core,{});


class neko_object320_oscr_no_route_base_lcam_ctc(neko_object320_oscr_no_route_factory):
    def mk_core(this, platform_cfg: neko_platform_cfg, params)->neko_object320_factory_core:
        return neko_object320_factory_core(platform_cfg, params);
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_wnaish_factory_cntr_lcam(this.core,{});

class neko_object320_oscr_no_route_base_lcam_lpos(neko_object320_oscr_no_route_base_lcam_ctc):
    def mkfeseqcls_head(this):
        return neko_lpos_roiseq_factoy(this.core,{});

class neko_object320_oscr_no_route_no_stn_asqz_factory(neko_object320_oscr_no_route_factory):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_factory_cntr_asqz(this.core,{});
class neko_object320_oscr_no_route_no_stn_lasqz_factory(neko_object320_oscr_no_route_factory):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_factory_cntr_lasqz(this.core,{});
    def mkfeseqcls_head(this):
        return neko_ctc_roiseq_factory(this.core,{});

# all correct will be forced to have 50% of total weight
class neko_object320_oscr_no_route_base_insseg_corrw05(neko_object320_oscr_no_route_base_lcam_ctc):
    def mksemseg_head(this):
        return neko_dese_val_semseg_factor_corr05(this.core, {});
    def mkfeseqcls_head(this):
        return None

class neko_object320_oscr_no_route_base_lcam_ctc_sem_corrw05(neko_object320_oscr_no_route_base_lcam_ctc):
    def mksemseg_head(this):
        return neko_dese_val_semseg_factor_corr05(this.core, {});


class neko_object320_oscr_no_route_base_lcam_ctc_sem_bga_corrw05(neko_object320_oscr_no_route_base_lcam_ctc_sem_corrw05):
    def mksemseg_head(this):
        return neko_dese_val_semseg_bga_factor_corr05(this.core, {});
class neko_object320_oscr_no_route_base_lcamXA_ctc_sem_corrw05(neko_object320_oscr_no_route_base_lcam_ctc_sem_corrw05):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_wnaish_factory_cntr_lcamXA(this.core, {});
class neko_object320_oscr_no_route_base_lcam_ctc_lsctrest_sem_corrw05(neko_object320_oscr_no_route_base_lcam_ctc_sem_corrw05):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack(this.core, {});

class neko_object320_oscr_no_route_base_lcam_lpos_lsctrest_sem_corrw05(neko_object320_oscr_no_route_base_lcam_ctc_lsctrest_sem_corrw05):
    def mkfeseqcls_head(this):
        return neko_lpos_roiseq_factoy(this.core,{});


class neko_object320_oscr_no_route_base_lcam_lpos_lsctrest_sem_corrw05_lsw(neko_object320_oscr_no_route_base_lcam_lpos_lsctrest_sem_corrw05):
    def mksemseg_head(this):
        return neko_dese_val_semseg_bga_factor_corr05(this.core, {});
    def mkca_insseg_head(this):
        return neko_first_gtlen_aux_cls_agnostic_insseg_factory(this.core,{});

class neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_sem_corrw05(neko_object320_oscr_no_route_base_lcam_ctc_lsctrest_sem_corrw05):
    def mk_core(this, platform_cfg: neko_platform_cfg, params) -> neko_object320_factory_core:
        return neko_object320_factory_core_nf(platform_cfg, params);

class neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_ind_sem_corrw05(neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_sem_corrw05):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_wnaish_factory_cntr_lcam_lsct_share_tfe_hack_ind(this.core, {});


class neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_sem_corrw05_nohack(neko_object320_oscr_no_route_base_lcam_ctc_sem_corrw05):
    def mk_core(this, platform_cfg: neko_platform_cfg, params) -> neko_object320_factory_core:
        return neko_object320_factory_core_nf(platform_cfg, params);
class neko_object320_oscr_no_route_norm_feat_sqz_ctc_lsctrest_sem_corrw05(neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_sem_corrw05):
    def mkfeseq_roiseq(this):
        return neko_object320_feseq_factory_cntr_sqz(this.core, {});


class neko_object320_oscr_no_route_norm_feat_lcam_lpos_lsctrest_sem_corrw05(neko_object320_oscr_no_route_norm_feat_lcam_ctc_lsctrest_sem_corrw05):
    def mkfeseqcls_head(this):
        return neko_lpos_roiseq_factoy(this.core,{});

