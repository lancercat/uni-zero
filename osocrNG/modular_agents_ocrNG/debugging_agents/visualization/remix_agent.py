import cv2
import numpy as np

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.debug.element_renderer import remix,flair_it
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent

# if we want to add routing information to flairs or log the .
class neko_remix_agent(neko_module_wrapping_agent):
    INPUT_all_patches="all_patches";
    INPUT_all_flairs="all_flairs";
    INPUT_raw_padded="raw_padded";
    INPUT_raw_image="raw_image";
    INPUT_aux_img_list="aux_img_list";
    OUTPUT_remixed_result="remixed";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.all_flairs = this.register_input_list(this.INPUT_all_flairs, iocvt_dict,"NEP_skipped_NEP");
        this.all_patches = this.register_input(this.INPUT_all_patches, iocvt_dict);
        this.aux_img_list = this.register_input_list(this.INPUT_aux_img_list, iocvt_dict);
        this.raw_image = this.register_input(this.INPUT_raw_image, iocvt_dict);
        this.remixed_results=this.register_output(this.OUTPUT_remixed_result,iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        aux_ims=[];
        for a in this.aux_img_list:
            if(a in workspace.inter_dict):
                aux_ims.append(workspace.get(a));
        if (this.raw_image in workspace.inter_dict):
            imgs_ = workspace.get(this.raw_image);
        else:
            imgs_=[_[0] for _ in aux_ims[0]];
            aux_ims=aux_ims[1:];
        aaps=[workspace.get(p) for p in this.all_patches];
        all_flairs = [workspace.get(f) for f in this.all_flairs];
        arim=[];
        for i in range(len(imgs_)):
            aa=[];
            for a in aux_ims:
                aa+=a[i];
            af=[];
            for f in all_flairs:
                af+=f[i];
            ap=[];
            for p in aaps:
                if(len(p[i])==0):
                    ap+=[np.zeros((32,32,3),dtype=np.uint8)]; # no prediction, just put a black box.
                else:
                    ap+=p[i];

            rim=remix(imgs_[i],aa,ap);
            if(len(af)):
                rim=flair_it(imgs_[i],rim,af);
            # cv2.imshow("meow",rim.astype(np.uint8));
            # cv2.waitKey(1);
            arim.append(rim);
        workspace.add(this.remixed_results,arim);
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,all_flairs,all_patches,aux_img_list,raw_image, raw_padded,remixed_result):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_all_flairs: all_flairs, cls.INPUT_all_patches: all_patches,
                           cls.INPUT_aux_img_list: aux_img_list, cls.INPUT_raw_image: raw_image,
                           cls.INPUT_raw_padded: raw_padded, cls.OUTPUT_remixed_result: remixed_result},
            "modcvt_dict": {}}}


def get_neko_remix_agent(
    all_flairs,all_patches,aux_img_list,raw_image,
    remixed_result
):
    engine = neko_remix_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_all_flairs: all_flairs, engine.INPUT_all_patches: all_patches, engine.INPUT_aux_img_list: aux_img_list, engine.INPUT_raw_image: raw_image, engine.OUTPUT_remixed_result: remixed_result}, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_remix_agent.print_default_setup_scripts();
