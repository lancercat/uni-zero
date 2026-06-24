import datetime
import os.path
import time

import cv2
import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.result_renderer import render_word
from osocrNG.names import default_ocr_variable_names as dvn
from osocrNG.modular_agents_ocrNG.output_logging_subs.acr_fps_reporter_hi import reporter_core,reporter_core_ARPF
class case_inv_acr_fps_reporter(neko_module_wrapping_agent):
    INPUT_pred_text_name=dvn.pred_text_name;
    INPUT_raw_label_name=dvn.raw_label_name;
    def reset(this,name,has_unk):
        if(not has_unk):
            this.recorder=reporter_core();
            # due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown
        else:
            this.recorder=reporter_core_ARPF();
        this.recorder.reset(name);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred_text=this.register(this.INPUT_pred_text_name,iocvt_dict,this.input_dict);
        this.gt_text=this.register(this.INPUT_raw_label_name,iocvt_dict,this.input_dict);

    def set_etc(this,param):
        pass;

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for gt,pr in zip(  workspace.inter_dict[this.gt_text],  workspace.inter_dict[this.pred_text]):
            this.recorder.record(gt, pr);
        # if (gt.lower() != pr.lower()):
        #     print(gt.lower(), "->", pr.lower());

    def report(this,environment:neko_environment):
        this.recorder.report(environment.epoch_idx,environment.batch_idx);


class case_inv_multi_orientation_acr_fps_reporter(neko_module_wrapping_agent):
    INPUT_pred_text_name=dvn.pred_text_name;
    INPUT_raw_label_name=dvn.raw_label_name;
    INPUT_raw_image_size_name=dvn.raw_image_size_name;
    PARAM_stat_range_dict="target_groups"
    def reset(this,name,has_unk):
        for k in this.stat_range_dict:
            if(not has_unk):
                this.recorders[k] = reporter_core();
            else:
                this.recorders[k]= reporter_core_ARPF();
        for rk in this.recorders:
            this.recorders[rk].reset(name+"_"+rk);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred_text=this.register(this.INPUT_pred_text_name,iocvt_dict,this.input_dict);
        this.gt_text=this.register(this.INPUT_raw_label_name,iocvt_dict,this.input_dict);
        this.raw_image_size=this.register(this.INPUT_raw_image_size_name,iocvt_dict,this.input_dict);

    def set_etc(this,param):
        this.stat_range_dict=neko_get_arg(
            this.PARAM_stat_range_dict,param,{"horizontal":{"ratio":(1.0,9999)},"vertical":{"ratio":(-9,1.0)},"all":{"ratio":(-9,9999)}}
        );
        this.recorders={};



        # due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for gt,pr,sz in zip(  workspace.inter_dict[this.gt_text],  workspace.inter_dict[this.pred_text],workspace.inter_dict[this.raw_image_size]):
            asr=sz[0]/sz[1];
            for k in this.recorders:
                if(asr>=this.stat_range_dict[k]["ratio"][0] and asr<this.stat_range_dict[k]["ratio"][1]):
                    this.recorders[k].record(gt,pr);
            # if (gt.lower() != pr.lower()):
            #     print(gt.lower(), "->", pr.lower());



    def report(this,environment:neko_environment):
        for k in this.recorders:
            this.recorders[k].report(environment.epoch_idx,environment.batch_idx);
