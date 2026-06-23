
from osocrNG.lsctNG.agents.mk1.mask_deformation import neko_trdg_mask_deformer_agent,neko_trdg_mask_deformer_agent_parallel
from osocrNG.lsctNG.agents.mk1.tdict_character_font_text_gen import neko_random_lsct_char_fnt_agent,neko_random_lsct_char_fnt_headless_agent

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.lsctNG.agents.mk1.mask_making import neko_mask_render_agent
from osocrNG.lsctNG.agents.mk1.random_bg import neko_random_bg_agent
from osocrNG.lsctNG.agents.mk1.fork_proto import lsctc_meta_fork
import os


from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN,P
from neko_sdk.cfgtool.argsparse import neko_get_arg

# activelearning gonna wait after we have blackwell together with better io. Prefetching is still the life line here. So we cannot afford gradient
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awa

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory

class neko_lsct_rand_char_factory(virtual_agt_factory):
    PARAM_seed="seed";
    PARAM_data_root="data_root";
    PARAM_orient="orientation";
    META_PRFX_local="meta_prfx";
    PARAM_fnt_root="fnt_root";
    PARAM_fnt_idx_file="fnt_idx";
    PARAM_im_root="imroot";
    PARAM_size="size";
    PARAM_charset="charset"
    PARAM_N="n";
    PARAM_K="k";
    def get_lsct_headless_agents(this, out_proto_raw_mask, out_utflabel, out_tdict,
                                 out_sample_raw_image,out_sample_text,out_sample_mask_transformed,out_sample_orient,
                                 inter_sample_raw_mask, inter_selected_utf,
                                 inter_fnts,inter_muxed_text,inter_muxed_mask, inter_muxed_orient,
                                 param_fnt_idx,param_fnt_root,param_bgroot,param_charset,param_k,param_N,param_size=64,seed=9):

        return {
        "agent": awa,
        "params": {
            "agent_list": ["make_label", "mask_gen","fork_proto", "transform_mask", "addbg"],
            "make_label": neko_random_lsct_char_fnt_headless_agent.get_agtcfg(
                inter_fnts,inter_selected_utf,inter_muxed_text,
                param_N,param_fnt_idx, param_fnt_root,param_k,param_charset,seed),
            "mask_gen": neko_mask_render_agent.get_agtcfg(inter_fnts, inter_muxed_text, inter_muxed_mask,  inter_muxed_orient,
                                                          [0], param_size),
            "fork_proto": lsctc_meta_fork.get_agtcfg(inter_muxed_mask,inter_selected_utf,out_utflabel,out_proto_raw_mask,out_sample_text,inter_sample_raw_mask,out_sample_orient,out_tdict,param_k),
            "transform_mask": neko_trdg_mask_deformer_agent.get_agtcfg(inter_sample_raw_mask, out_sample_orient,
            out_sample_mask_transformed,
            param_size),
            "addbg": neko_random_bg_agent.get_agtcfg( out_sample_mask_transformed,  out_sample_raw_image, param_bgroot, param_k)
            }
        };


    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        size=neko_get_arg(this.PARAM_size,params,32);

        fnt_idx_file = neko_get_arg(this.PARAM_fnt_idx_file, params);
        fnt_root = neko_get_arg(this.PARAM_fnt_root, params);
        imroot=neko_get_arg(this.PARAM_im_root,params);
        k=neko_get_arg(this.PARAM_K,params,2);
        n = neko_get_arg(this.PARAM_N, params, 64);
        seed =neko_get_arg(this.PARAM_seed,params,9);
        lsct_data_prfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);
        lsct_meta_prfx = neko_get_arg(this.META_PRFX_local, data_prfx_dict);
        inter_prfx=P(lsct_data_prfx,"inter");
        charset=neko_get_arg(this.PARAM_charset,params);
        agtcfg=awa.append_agent_to_cfg(agtcfg,"lsct_maker",this.get_lsct_headless_agents(
            VN.IM_raw(lsct_meta_prfx),VN.UTF(lsct_meta_prfx),VN.TDICT(lsct_meta_prfx),
            VN.IM_raw(lsct_data_prfx),VN.UTF(lsct_data_prfx),VN.GT_BIN_MSK(lsct_data_prfx),VN.SAM_ORIENT(lsct_data_prfx),
            VN.GT_BIN_MSK(inter_prfx),VN.DBG_SEL_UTF(inter_prfx),VN.DBG_SELECTED_FNT(inter_prfx),VN.DBG_MUX_UTF(inter_prfx),
            VN.DBG_MUX_MSK(inter_prfx),VN.DBG_MUX_ORI(inter_prfx),
            fnt_idx_file, fnt_root, imroot,charset,k,n,size,seed=seed
        )); # the new framework requires the data loader to tokenize.
        return agtcfg;
    #        pass;
