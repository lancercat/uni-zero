# from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
# from neko_sdk.neko_framework_NG.libimfe.agents.unified_im_fe_agent import imfe_NG_sub
# from neko_sdk.neko_framework_NG.agents.sampling.aggr_one import aggrone
# from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_simple_action_module_wrapping_agent_1i1o_basic
# from neko_sdk.neko_framework_NG.agents.utils.fetch_from_list import neko_fetch_from_list
# from neko_sdk.neko_framework_NG.agents.utils.detach_agent import neko_detach_agent
# from neko_sdk.neko_framework_NG.names import P
# from neko_sdk.neko_framework_NG.libimfe.agents.map2vec import neko_simple_map_tower_to_vec_aaggr_agent_builder
# # To do: Deprecated
# class neko_im_tensor_to_feat_agent:
#
#     @classmethod
#     def arm_fe(cls,acfg,tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, mod_fe_name):
#
#         acfg = awa.append_agent_to_cfg(acfg, "fe", imfe_NG_sub.get_agtcfg(
#              tensor_img_name,
#             tensor_featmap_name,feature_tower,feature_tower_tag,
#             mod_fe_name
#         ));
#         return acfg
#     @classmethod
#     def arm_sampling(cls,acfg,
#                    tensor_featmap_name,feature_tower, tensor_att_name,
#                    mod_se_name,mod_att_name):
#
#         local_temp_feat_name=P(tensor_featmap_name,"temp_feat");
#         local_temp_feat_se_name=P(local_temp_feat_name,"spatial_embedded");
#         acfg=awa.append_agent_to_cfg(acfg,"pick_tf",neko_fetch_from_list.get_agtcfg(
#             feature_tower,local_temp_feat_name,0
#         ));
#         acfg = awa.append_agent_to_cfg(acfg, "attn", neko_simple_action_module_wrapping_agent_1i1o_basic.get_agtcfg(
#             local_temp_feat_name, tensor_att_name, mod_att_name
#         ));
#         return acfg
#     @classmethod
#     def arm_core_agtcfg(cls,acfg,
#                    tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, tensor_att_name, tensor_vec_name,
#                    mod_fe_name,mod_se_name,mod_att_name,  mod_drop_name,
#                    ):
#         acfg=cls.arm_fe(acfg,tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, mod_fe_name);
#         acfg= cls.arm_sampling(acfg,
#                    tensor_featmap_name,feature_tower, tensor_att_name,
#                    mod_se_name,mod_att_name)
#         acfg = awa.append_agent_to_cfg(acfg, "aggr", aggrone.get_agtcfg(
#             tensor_att_name, tensor_featmap_name, tensor_vec_name, mod_drop_name
#         ));
#         return acfg;
#     @classmethod
#     def get_agtcfg(cls,
#                    tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, tensor_att_name, tensor_vec_name,
#                    mod_fe_name,mod_se_name,mod_att_name,  mod_drop_name,
#                    ):
#         acfg=awa.empty();
#         acfg=cls.arm_core_agtcfg(acfg, tensor_img_name,
#                                  tensor_featmap_name, feature_tower, feature_tower_tag,
#                                  tensor_att_name, tensor_vec_name,
#                                  mod_fe_name, mod_se_name, mod_att_name, mod_drop_name,
#                                  );
#         return acfg;
#
#
# class neko_simple_im_tensor_to_feat_agent:
#
#     @classmethod
#     def arm_fe(cls,acfg,tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, mod_fe_name):
#
#         acfg = awa.append_agent_to_cfg(acfg, "fe", imfe_NG_sub.get_agtcfg(
#              tensor_img_name,
#             tensor_featmap_name,feature_tower,feature_tower_tag,
#             mod_fe_name
#         ));
#         return acfg
#     @classmethod
#     def arm_sampling(cls,acfg,
#                    tensor_featmap_name,feature_tower, tensor_att_name,
#                    mod_att_name):
#
#         local_temp_feat_name=P(tensor_featmap_name,"temp_feat");
#         local_temp_feat_se_name=P(local_temp_feat_name,"spatial_embedded");
#         acfg=awa.append_agent_to_cfg(acfg,"pick_tf",neko_fetch_from_list.get_agtcfg(
#             feature_tower,local_temp_feat_name,0
#         ));
#         acfg = awa.append_agent_to_cfg(acfg, "attn", neko_simple_action_module_wrapping_agent_1i1o_basic.get_agtcfg(
#             local_temp_feat_name, tensor_att_name, mod_att_name
#         ));
#         return acfg
#     @classmethod
#     def arm_core_agtcfg(cls,acfg,
#                    tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, tensor_att_name, tensor_vec_name,
#                    mod_fe_name,mod_att_name,  mod_drop_name,
#                    ):
#         acfg=cls.arm_fe(acfg,tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, mod_fe_name);
#         acfg= cls.arm_sampling(acfg,
#                    tensor_featmap_name,feature_tower, tensor_att_name,
#                    mod_att_name)
#         acfg = awa.append_agent_to_cfg(acfg, "aggr", aggrone.get_agtcfg(
#             tensor_att_name, tensor_featmap_name, tensor_vec_name, mod_drop_name
#         ));
#         return acfg;
#     @classmethod
#     def get_agtcfg(cls,
#                    tensor_img_name,
#                    tensor_featmap_name,feature_tower,feature_tower_tag, tensor_att_name, tensor_vec_name,
#                    mod_fe_name,mod_att_name,  mod_drop_name,
#                    ):
#         acfg=awa.empty();
#         acfg=cls.arm_core_agtcfg(acfg, tensor_img_name,
#                                  tensor_featmap_name, feature_tower, feature_tower_tag,
#                                  tensor_att_name, tensor_vec_name,
#                                  mod_fe_name, mod_att_name, mod_drop_name,
#                                  );
#         return acfg;
#
#
