import cv2
import numpy as np

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from osocrNG.lsctNG.agents.mk1.random_text_font_gen_grounded import neko_random_grounded_font_fetching_agent,neko_random_font_repeater,neko_random_font_shuff
from osocrNG.lsctNG.agents.mk1.mask_making import neko_mask_render_agent_insmask
from osocrNG.lsctNG.agents.mk1.mask_deformation import neko_trdg_mask_deformer_agent_insmsk
from osocrNG.lsctNG.agents.mk1.random_bg import neko_random_bg_agent
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

# lsctw med trdg engine.
class lsct_shadow_data_agent_factory:
    INPUT_text_label="text";
    @classmethod
    def get_lsct_core(cls,in_text_label,
                          out_fnt_choices,out_tf_dict,out_syn_text,out_syn_fnt,out_syn_mask,out_syn_mask_ins,out_syn_orient,out_syn_mask_trans,out_syn_mask_ins_trans,out_syn_ins_mask_utf,out_raw_images,
                          param_fnt_idx_file,param_fnt_root,param_imroot,param_orient,param_size,chaspace,text_making_agent):

        if chaspace is None:
            chaspace=[0,32,64];
        ac={
            "agent": awa,
            "params": {
                "agent_list": ["selfntcand","gen_text","gen_mask","transform_mask","addbg"],
                "selfntcand": neko_random_grounded_font_fetching_agent.get_agtcfg(in_text_label,out_fnt_choices,out_tf_dict,param_fnt_idx_file,param_fnt_root,3),
                "gen_text" :text_making_agent,
                "gen_mask" : neko_mask_render_agent_insmask.get_agtcfg(out_syn_fnt, out_syn_text, out_syn_mask,out_syn_mask_ins,out_syn_ins_mask_utf, out_syn_orient, param_orient, param_size,chaspace),
                "transform_mask": neko_trdg_mask_deformer_agent_insmsk.get_agtcfg(out_syn_mask,out_syn_mask_ins,out_syn_orient,out_syn_mask_trans,out_syn_mask_ins_trans,param_size),
                "addbg": neko_random_bg_agent.get_agtcfg(out_syn_mask_trans,out_raw_images,param_imroot,3),
            }
        }
        return ac;
    @classmethod
    def get_lsct_repeater(cls,in_text_label,
                          out_fnt_choices,out_tf_dict,out_syn_text,out_syn_fnt,out_syn_mask,out_syn_mask_ins,
                          out_syn_orient,out_syn_mask_trans,out_syn_mask_ins_trans,
                          out_syn_ins_mask_utf,
                          out_raw_images,
                          param_fnt_idx_file,param_fnt_root,param_imroot,param_orient,param_size,chaspace):
        text_making_agent= neko_random_font_repeater.get_agtcfg(out_fnt_choices, in_text_label, out_tf_dict, out_syn_fnt, out_syn_text);
        return cls.get_lsct_core(in_text_label,
                          out_fnt_choices,out_tf_dict,out_syn_text,out_syn_fnt,out_syn_mask,out_syn_mask_ins,out_syn_orient,out_syn_mask_trans,out_syn_mask_ins_trans,out_syn_ins_mask_utf,out_raw_images,
                          param_fnt_idx_file,param_fnt_root,param_imroot,param_orient,param_size,chaspace,text_making_agent,

                );


    @classmethod
    def get_lsct_shuf(cls,in_text_label,
                          out_fnt_choices,out_tf_dict,out_syn_text,out_syn_fnt,out_syn_mask,out_syn_mask_ins,out_syn_orient,out_syn_mask_trans,out_syn_mask_ins_trans,out_syn_ins_mask_utf,out_raw_images,
                          param_fnt_idx_file,param_fnt_root,param_imroot,param_orient,param_size,chaspace):
        text_making_agent = neko_random_font_shuff.get_agtcfg(out_fnt_choices, in_text_label, out_tf_dict, out_syn_fnt, out_syn_text);
        return cls.get_lsct_core(in_text_label,
                                 out_fnt_choices, out_tf_dict, out_syn_text, out_syn_fnt, out_syn_mask,
                                 out_syn_mask_ins, out_syn_orient, out_syn_mask_trans, out_syn_mask_ins_trans,out_syn_ins_mask_utf,
                                 out_raw_images,
                                 param_fnt_idx_file, param_fnt_root, param_imroot, param_orient, param_size, chaspace,
                                 text_making_agent,

                                 );

if __name__ == '__main__':
    f=lsct_shadow_data_agent_factory();

    param_fnt_idx_file = "/home/lasercat/ssddata/synth_lsct/cache/finfo251105.pt";
    param_fnt_root = "/home/lasercat/ssddata/synth_lsct/";

    param_orient = [0, 1];
    param_imroot = "/home/lasercat/ssddata/synth_lsct/bg_img/"
    param_size = 64;

    in_text_label="text";

    out_fnt_choices="fnt_choices";
    out_tf_dict="token_fnt_dict";
    out_syn_fnt="synth_fnt";
    out_syn_mask="synth_mask_raw";
    out_syn_mask_ins="synth_mask_ins";

    # endpoints
    out_raw_images="out_raw_images";
    out_syn_text = "synth_text";
    out_syn_mask_trans="synth_text_mask";
    out_syn_mask_orient="synth_text_orientation";
    out_syn_mask_ins_trans ="synth_text_mask_inst"
    ac=f.get_lsct_shuf(
        in_text_label,
        out_fnt_choices, out_tf_dict, out_syn_text, out_syn_fnt, out_syn_mask, out_syn_mask_ins, out_syn_mask_orient,
        out_syn_mask_trans, out_syn_mask_ins_trans, out_raw_images,
        param_fnt_idx_file, param_fnt_root, param_imroot, param_orient, param_size, [0]
    )

    a=ac["agent"](ac["params"]);

    from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
    w=neko_workspace();
    e=neko_environment();
    w.add(in_text_label,["ድመት","बिल्ली","cat","ねこ","猫","кот","кішка","고양이"]);
    a.take_action(w,e);
    for i in range(len(w.inter_dict[out_raw_images])):
        all=[w.inter_dict[out_raw_images][i],(w.inter_dict[out_syn_mask_trans][i]/255*w.inter_dict[out_raw_images][i]).astype(np.uint8)];
        for j in range(len(tokenize(w.inter_dict[in_text_label][i]))):
            all.append((w.inter_dict[out_syn_mask_ins_trans][i][j]/255*w.inter_dict[out_raw_images][i]).astype(np.uint8))
        if(all[0].shape[0]>all[0].shape[1]):
            cv2.imshow("meow"+str(i),np.concatenate(all,axis=1));
        else:
            cv2.imshow("meow"+str(i), np.concatenate(all, axis=0));
    cv2.waitKey(0);

