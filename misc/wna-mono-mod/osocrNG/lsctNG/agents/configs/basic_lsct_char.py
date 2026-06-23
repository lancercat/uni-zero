
from osocrNG.lsctNG.agents.mk1.mask_deformation import neko_trdg_mask_deformer_agent,neko_trdg_mask_deformer_agent_parallel
from osocrNG.lsctNG.agents.mk1.tdict_character_font_text_gen import neko_random_lsct_char_fnt_agent,neko_random_lsct_char_fnt_headless_agent

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.lsctNG.agents.mk1.mask_making import neko_mask_render_agent,neko_mask_render_agent_parallel
from osocrNG.lsctNG.agents.mk1.random_bg import neko_random_bg_agent
from osocrNG.lsctNG.agents.mk1.fork_proto import lsctc_meta_fork

class lsct_char_data_agent_factory:
    @classmethod
    def get_lsct_data_agents(this, in_selected_utf,
                             dev_ref, out_fnts,
                             out_text, char_raw_mask, char_orient, char_mask_transformed, char_raw_image, param_fnt_idx,param_fnt_root,param_bgroot):
        return {
        "agent": awa,
        "params": {
            "agent_list": ["make_label", "mask_gen", "transform_mask", "addbg"],
            "make_label": neko_random_lsct_char_fnt_agent.get_agtcfg(in_selected_utf, dev_ref, out_fnts, out_text,
                                                                     param_fnt_idx, param_fnt_root, 1
                                                                     ),
            "mask_gen": neko_mask_render_agent.get_agtcfg(out_fnts, out_text, char_raw_mask, char_orient,
                                                          [0], 64),
        "transform_mask": neko_trdg_mask_deformer_agent.get_agtcfg(char_raw_mask,char_orient,
            char_mask_transformed,
            64),
        "addbg": neko_random_bg_agent.get_agtcfg(char_mask_transformed, char_raw_image, param_bgroot, 1)
            }
        };

    # @classmethod
    # def get_lsct_headless_agents(this, out_proto_raw_mask, out_utflabel, out_tdict,
    #                              out_sample_raw_image,out_sample_text,out_sample_mask_transformed,out_sample_orient,
    #                              inter_sample_raw_mask, inter_selected_utf,
    #                              inter_fnts,inter_muxed_text,inter_muxed_mask, inter_muxed_orient,
    #                              param_fnt_idx,param_fnt_root,param_bgroot,param_k=2,seed=9):
    #
    #     return {
    #     "agent": awa,
    #     "params": {
    #         "agent_list": ["make_label","mask_gen"],
    #         "make_label": neko_random_lsct_char_fnt_headless_agent.get_agtcfg(
    #             inter_fnts,inter_selected_utf,inter_muxed_text,
    #             512,param_fnt_idx, param_fnt_root,param_k,seed),
    #         "mask_gen": neko_mask_render_agent.get_agtcfg(inter_fnts, inter_muxed_text, inter_muxed_mask,  inter_muxed_orient,
    #                                                        [0], 64),
    #                    }
    #     };
    @classmethod
    def get_lsct_headless_agents(this, out_proto_raw_mask, out_utflabel, out_tdict,
                                 out_sample_raw_image,out_sample_text,out_sample_mask_transformed,out_sample_orient,
                                 inter_sample_raw_mask, inter_selected_utf,
                                 inter_fnts,inter_muxed_text,inter_muxed_mask, inter_muxed_orient,
                                 param_fnt_idx,param_fnt_root,param_bgroot,param_charset,param_k=2,param_N=320,seed=9):

        return {
        "agent": awa,
        "params": {
            "agent_list": ["make_label", "mask_gen","fork_proto", "transform_mask", "addbg"],
            "make_label": neko_random_lsct_char_fnt_headless_agent.get_agtcfg(
                inter_fnts,inter_selected_utf,inter_muxed_text,
                param_N,param_fnt_idx, param_fnt_root,param_k,param_charset,seed),
            "mask_gen": neko_mask_render_agent_parallel.get_agtcfg(inter_fnts, inter_muxed_text, inter_muxed_mask,  inter_muxed_orient,
                                                          [0], 64),
            "fork_proto": lsctc_meta_fork.get_agtcfg(inter_muxed_mask,inter_selected_utf,out_utflabel,out_proto_raw_mask,out_sample_text,inter_sample_raw_mask,out_sample_orient,out_tdict,param_k),
            "transform_mask": neko_trdg_mask_deformer_agent.get_agtcfg(inter_sample_raw_mask, out_sample_orient,
            out_sample_mask_transformed,
            64),
            "addbg": neko_random_bg_agent_parallel.get_agtcfg( out_sample_mask_transformed,  out_sample_raw_image, param_bgroot, param_k)
            }
        };
