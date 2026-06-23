from neko_sdk.neko_framework_NG.names import P

from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.result_rendering import neko_result_rendering_agent_via_utf
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.attention_visualization import \
    get_attention_visualization_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.matching_flair_renders import \
    get_neko_perfect_match_flair_making_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.tensor_im_visualization import \
    neko_tensor_im_visualization
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.remix_agent import neko_remix_agent
from osocrNG.modular_agents_ocrNG.debugging_agents.visualization.combine_batch_remix_agent import neko_combined_show_agent


from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,global_mod_cfg
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

class debug_vis_agent_factory(virtual_agt_factory):

    def get_lpos_att_vis(this,task_data_prefix, img_data_prefix,attn_data_prefix, armed_meta_prfx,raw_meta_prfx,debug_prefix):
        return get_attention_visualization_agent(VN.WORD_ATT_MAP(attn_data_prefix),
                                                            VN.IM_normed_tensor(attn_data_prefix),
                                                            VN.WORD_LEN_PRED_VALUE(attn_data_prefix),
                                                            VN.DBG_ATT_IMS(debug_prefix));
    def get_result_gt_renderer(this,target_meta_prefix,pred_data_prefix,gt_data_prfx,debug_prefix):
        return neko_result_rendering_agent_via_utf.get_agtcfg(
            VN.PRED_TOK(pred_data_prefix),VN.GT_TOK_UTF_WUNK(gt_data_prfx),
            VN.UTF(target_meta_prefix),VN.TDICT(target_meta_prefix),
            VN.IM_normed_tensor(target_meta_prefix),VN.DBG_GT_VPR_PATCHES(debug_prefix),"");
    def get_neko_tensor_im_visualization(this,data_data_prefix,debug_prfx):
        return neko_tensor_im_visualization.get_agtcfg(VN.IM_normed_tensor(data_data_prefix),VN.DBG_IMTEN(debug_prfx),127,127);
    # def get_neko_remix_agent_im_gt_pred(this,endpoint_data_prefix,debug_prefix):
    #     return neko_remix_agent.get_agtcfg(
    #         [],
    #         [VN.DBG_GT_VPR_PATCHES(debug_prefix)],
    #         [VN.DBG_IMTEN(debug_prefix)],
    #         VN.IM_normed_tensor(endpoint_data_prefix),
    #         "NEP_skipped_NEP",
    #         VN.DBG_SMPRNDR(debug_prefix)
    #     );
    def get_neko_batch_show_agent(this,debug_prefix):
        return neko_combined_show_agent.get_agtcfg(
            VN.DBG_SMPRNDR(debug_prefix),VN.DBG_CANVAS(debug_prefix),winname=debug_prefix
        );
    def arm_320_char_debugger(this,target_meta_prefix,img_tensor_data_prfx,pred_data_prefix,gt_data_prfx,debug_prefix,acfg,aprfx):
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"result_rndr"),this.get_result_gt_renderer(target_meta_prefix, pred_data_prefix, gt_data_prfx, debug_prefix))
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"tensor_vis"),this.get_neko_tensor_im_visualization(img_tensor_data_prfx,debug_prefix));
        acfg = awa.append_agent_to_cfg(acfg, P(aprfx, "compile"), neko_remix_agent.get_agtcfg(
            [],
            [VN.DBG_GT_VPR_PATCHES(debug_prefix)],
            [VN.DBG_IMTEN(debug_prefix)],
            "NEP_this_does_not_exist_NEP", # dfake it
            "NEP_skipped_NEP",
            VN.DBG_SMPRNDR(debug_prefix)
        ));
        acfg=awa.append_agent_to_cfg(acfg,P(aprfx,"show"),this.get_neko_batch_show_agent(debug_prefix));
        return acfg;


        # def get_result_vis(this,task_data_prefix, feat_data_prefix, armed_meta_prfx,raw_meta_prfx):
    #     ac = {
    #         "agent": neko_agent_wrapping_agent,
    #         "params": {
    #             "agent_list": ["visatt", "vis_tensor_im", "correctness_flair", "routing_flair", "render", "remix"],
    #             "visatt":
    #             "vis_tensor_im": get_neko_tensor_im_visualization(VN..TEN_IMG_NAME,
    #                                                               VN..DBG_PADDED_RAW_IM, 127, 127),
    #             "correctness_flair": get_neko_perfect_match_flair_making_agent(
    #                 VN..PRED_TEXT, VN..RAW_LABEL_NAME,
    #                 VN..DBG_MATCHING_FLAIRS),
    #             "routing_flair": get_neko_routing_flair_making_agent(gprfix + VN..ROUTER_ACT_NAME,
    #                                                                  VN..DBG_ROUTING_FLAIRS,
    #                                                                  "NEP_default_NEP"),
    #             "render": get_neko_result_rendering_agent(
    #                 gprfix + this.VAN.PROTO_LABEL, VN..PRED_TEXT,
    #                 gprfix + this.VAN.TDICT, gprfix + this.VAN.TENSOR_PROTO_IMG_NAME,
    #                 VN..RAW_LABEL_NAME, VN..DBG_GT_VPR_PATCHES,
    #                 "NEP_default_NEP"),
    #             "remix": get_neko_remix_agent(
    #                 [VN..DBG_MATCHING_FLAIRS, VN..DBG_ROUTING_FLAIRS],
    #                 [VN..DBG_GT_VPR_PATCHES],
    #                 [VN..DBG_PADDED_RAW_IM,
    #                  VN..DBG_ATT_IMS], VN..RAW_IMG_NAME,
    #                                                  VN..DBG_RESULT_PANEL),
    #             neko_agent_wrapping_agent.PARAM_ACT_VARS: [VN..RAW_IMG_NAME]
    #         },
    #     }
    #     return ac;
