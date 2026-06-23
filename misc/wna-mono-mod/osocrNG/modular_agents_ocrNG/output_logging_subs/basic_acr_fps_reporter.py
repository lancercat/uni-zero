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
from neko_sdk.ocr_modules.sptokens import tUNKREP

from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

class case_inv_acr_fps_reporter(neko_module_wrapping_agent):
    INPUT_pred_text_name=dvn.pred_text_name;
    INPUT_raw_label_name=dvn.raw_label_name;
    INPUT_tdict_name=dvn.tdict_name;
    PARAM_variant="variant"
    def reset(this,name,has_unk):
        if(not has_unk):
            this.recorder=reporter_core();# due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown
        else:
            this.recorder=reporter_core_ARPF();

        this.recorder.reset(name+this.variant);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred_text=this.register(this.INPUT_pred_text_name,iocvt_dict,this.input_dict);
        this.gt_text=this.register(this.INPUT_raw_label_name,iocvt_dict,this.input_dict);
        this.tdict=this.register(this.INPUT_tdict_name,iocvt_dict,this.input_dict,dvn.tdict);
    def set_etc(this,param):
        this.variant=neko_get_arg(this.PARAM_variant,param,"");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(this.tdict in workspace.inter_dict):
            tdict=workspace.inter_dict[this.tdict];
        else:
            tdict=None;
        if(this.gt_text not in workspace.inter_dict):
            return; # no gt, no report

        for gt,pr in zip(  workspace.inter_dict[this.gt_text],  workspace.inter_dict[this.pred_text]):
            if(tdict is not None):
                l=[];
                for c in tokenize(gt):
                    if(c not in tdict):
                        l.append(tUNKREP);
                    else:
                        l.append(c);
                gt_="".join(l);
            else:
                gt_=gt;
            this.recorder.record(gt_, pr);

        # if (gt.lower() != pr.lower()):
        #     print(gt.lower(), "->", pr.lower());

    def report(this,environment:neko_environment):
        rpt=this.recorder.report(environment.epoch_idx,environment.batch_idx);
        return rpt;


class case_inv_multi_orientation_acr_fps_reporter(neko_module_wrapping_agent):
    INPUT_pred_text_name=dvn.pred_text_name;
    INPUT_raw_label_name=dvn.raw_label_name;
    INPUT_raw_image_size_name=dvn.raw_image_size_name;
    INPUT_tdict_name=dvn.tdict_name;
    PARAM_stat_range_dict="target_groups"
    def reset(this,name,has_unk):
        for k in this.stat_range_dict:
            if (not has_unk):
                this.recorders[k] = reporter_core();
            else:
                this.recorders[k] = reporter_core_ARPF();

        for rk in this.recorders:
            this.recorders[rk].reset(name+"_"+rk);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred_text=this.register(this.INPUT_pred_text_name,iocvt_dict,this.input_dict);
        this.tdict=this.register(this.INPUT_tdict_name,iocvt_dict,this.input_dict,dvn.tdict);
        this.gt_text=this.register(this.INPUT_raw_label_name,iocvt_dict,this.input_dict);
        this.raw_image_size=this.register(this.INPUT_raw_image_size_name,iocvt_dict,this.input_dict);

    def set_etc(this,param):
        this.stat_range_dict=neko_get_arg(
            this.PARAM_stat_range_dict,param,{"horizontal":{"ratio":(1.0,9999)},"vertical":{"ratio":(-9,1.0)},"all":{"ratio":(-9,9999)}}
        );
        this.recorders={};


        # due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        if(this.tdict in workspace.inter_dict):
            tdict=workspace.inter_dict[this.tdict];
        else:
            tdict=None;
        for gt,pr,sz in zip(  workspace.inter_dict[this.gt_text],  workspace.inter_dict[this.pred_text],workspace.inter_dict[this.raw_image_size]):
            if(tdict is not None):
                l=[];
                for c in gt:
                    if(c not in tdict):
                        l.append(tUNKREP);
                    else:
                        l.append(c);
                gt_="".join(l);
            else:
                gt_=gt;

            asr=sz[0]/sz[1];
            for k in this.recorders:
                if(asr>=this.stat_range_dict[k]["ratio"][0] and asr<this.stat_range_dict[k]["ratio"][1]):
                    this.recorders[k].record(gt_,pr);
            if (gt.lower() != pr.lower()):
                print(gt.lower(), "->", pr.lower());



    def report(this,environment:neko_environment):
        for k in this.recorders:
            this.recorders[k].report(environment.epoch_idx,environment.batch_idx);


class neko_debugging_image_pred_logging_reporter(neko_module_wrapping_agent):
    PARAM_save_path="save_path";
    PARAM_test_meta_path="meta_path";
    PARAM_show_name="show_name";

    def set_etc(this,param):
        this.export_path=param[this.PARAM_save_path];
        try:
            this.mdict=torch.load(param[this.PARAM_test_meta_path]);
        except:
            this.mdict=None;

        this.show=neko_get_arg(this.PARAM_show_name,param,"NEP_skipped_NEP");
        this.wait=neko_get_arg("wait",param,30);
        this.padvalue=neko_get_arg("padvalue",param,127)
        this.cntr=0;
        this.testname="nep";
    def reset(this,test_name):
        this.cntr = 0;
        this.testname=test_name;

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.pred_text=neko_get_arg("pred_text",iocvt_dict,"pred_text");
        this.input_dict.gt_text=neko_get_arg("text_label",iocvt_dict,"text_label");
        this.input_dict.image=neko_get_arg("raw_image",iocvt_dict,"raw_image");
        this.input_dict.tdict=neko_get_arg("tdict",iocvt_dict,"tdict");
        this.input_dict.plabel=neko_get_arg("plabel",iocvt_dict,"plabel");

        this.input_dict.tensor_proto_img=neko_get_arg("tensor_proto_img",iocvt_dict,"tensor_proto_img");
        this.input_dict.selection=neko_get_arg("selected_head",iocvt_dict,"selected_head");
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        imgs_=workspace.inter_dict[this.input_dict.image];
        if(imgs_[0].shape[0]>imgs_[0].shape[1]):
            imgs=[i.transpose([1,0,2]) for i in imgs_];
        else:
            imgs=imgs_;

        for i in range(len(imgs)):
            this.cntr += 1;
            if this.mdict is not None:
                md=this.mdict;
            else:
                pimg=workspace.inter_dict[this.input_dict["tensor_proto_img"]];
                plab=workspace.inter_dict[this.input_dict["plabel"]];
                apimg=pimg+0;
                for j in range(pimg.shape[0]):
                    apimg[plab[j].item()]=pimg[j];
                apimg=apimg[:plab.max().item()];

                md={
                    "label_dict":workspace.inter_dict[this.input_dict["tdict"]],
                    "protos":(apimg*127+127).unsqueeze(1).cpu().to(torch.uint8)
                }
            if(this.input_dict.pred_text in workspace.inter_dict):
                rim,_=render_word(
                    md,latin62.union(t1_3755),imgs[i],
                    workspace.inter_dict[this.input_dict.gt_text][i].lower(),
                    workspace.inter_dict[this.input_dict.pred_text][i].lower(),padvalue=this.padvalue);
            else:
                rim,_=render_word(md,latin62.union(t1_3755),imgs[i],
                                  workspace.inter_dict[this.input_dict.gt_text][i].lower(),
                                  None,padvalue=this.padvalue);
            cv2.imwrite(os.path.join(this.export_path,str(this.cntr)+".jpg"),rim);
            if(this.show is not None):
                cv2.namedWindow(this.show,0);
                cv2.imshow(this.show,rim);
                cv2.waitKey(this.wait);

    def report(this,environment:neko_environment):
        pass;
