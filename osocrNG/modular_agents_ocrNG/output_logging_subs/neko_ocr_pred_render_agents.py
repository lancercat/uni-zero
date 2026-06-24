import numpy as np

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.ocr_modules.result_renderer import render_word
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62
import torch
import cv2
import os
#
class neko_image_pred_render_agent(neko_module_wrapping_agent):
    PARAM_test_meta_path="meta_path";
    PARAM_pad_value="padvalue";
    INPUT_pred_text="pred_text";
    INPUT_text_label="text_label";
    INPUT_raw_image="raw_image";
    INPUT_tdict="tdict";
    INPUT_proto_label="plabel";
    INPUT_tensor_proto_img = "tensor_proto_img";
    OUTPUT_rendered_result = "rendered_result";
    def reset(this,*param):
        pass;
    def report(this,*param):
        pass;

    def set_etc(this,param):
        try:
            this.mdict=torch.load(param[this.PARAM_test_meta_path]);
        except:
            this.mdict=None;

        this.padvalue=neko_get_arg(this.PARAM_pad_value,param,127)


    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred_text=this.register(this.INPUT_pred_text,iocvt_dict,this.input_dict);
        this.text_label=this.register(this.INPUT_text_label, iocvt_dict, this.input_dict,"NEP_skipped_NEP");
        this.raw_image=this.register(this.INPUT_raw_image, iocvt_dict, this.input_dict);
        this.tdict=this.register(this.INPUT_tdict,iocvt_dict,this.input_dict);
        this.plabel=this.register(this.INPUT_proto_label,iocvt_dict,this.input_dict);
        this.proto_img=this.register(this.INPUT_tensor_proto_img,iocvt_dict,this.input_dict);

        this.rendered_results= this.register(this.OUTPUT_rendered_result,iocvt_dict,this.output_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        imgs=workspace.inter_dict[this.raw_image];
        if this.mdict is not None:
            md = this.mdict;
        else:
            pimg = workspace.inter_dict[this.proto_img];
            plab = workspace.inter_dict[this.plabel];
            apimg = pimg + 0;
            haz=set();
            for j in range(pimg.shape[0]):
                if(plab[j].item() in haz):
                    continue;
                haz.add(plab[j].item());
                apimg[plab[j].item()] = pimg[j];
            apimg = apimg[:plab.max().item()];
            md = {
                "label_dict": workspace.inter_dict[this.tdict],
                "protos": (apimg * 127 + 127).unsqueeze(1).cpu().to(torch.uint8)
            }
        tarlist=[];
        for i in range(len(imgs)):
            vert=False;

            if (imgs[i].shape[0] > imgs[i].shape[1]):
                vert=True;
            img = cv2.cvtColor(np.array(imgs[i]),cv2.COLOR_RGB2BGR);


            if(this.text_label in workspace.inter_dict):
                rim,_=render_word(
                    md,latin62.union(t1_3755),img,
                    workspace.inter_dict[this.text_label][i].lower(),
                    workspace.inter_dict[this.pred_text][i].lower(),padvalue=this.padvalue,transposed=vert);
            else:
                rim,_=render_word(md,latin62.union(t1_3755),img,
                                 None,
                                  workspace.inter_dict[this.pred_text][i],padvalue=this.padvalue,transposed=vert);
            tarlist.append(rim);
        workspace.inter_dict[this.rendered_results]=tarlist;

def get_img_render_agent(pred_text,text_label,raw_image,tdict,plabel,tensor_proto_img,rendered_result,dict_path=None):
    return {
            "agent":neko_image_pred_render_agent,
            "params":{
                neko_image_pred_render_agent.PARAM_pad_value:127,
                neko_image_pred_render_agent.PARAM_test_meta_path: None,
                "iocvt_dict":{
                    neko_image_pred_render_agent.INPUT_pred_text:pred_text,
                    neko_image_pred_render_agent.INPUT_text_label: text_label,
                    neko_image_pred_render_agent.INPUT_raw_image: raw_image,
                    neko_image_pred_render_agent.INPUT_tdict: tdict,
                    neko_image_pred_render_agent.INPUT_proto_label: plabel,
                    neko_image_pred_render_agent.INPUT_tensor_proto_img: tensor_proto_img,
                    neko_image_pred_render_agent.OUTPUT_rendered_result: rendered_result,
                },
                "modcvt_dict": {

                }
            }
    };