import glob
import os.path
import random

import PIL.Image
import numpy as np

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.lsctNG.libtrdg.functional import mix_with_gen_bg_serial
import cv2
from PIL import Image, ImageFilter, ImageColor
from multiprocess.pool import Pool


class neko_random_bg_agent(neko_module_wrapping_agent):
    INPUT_masks_np="masks_np"; # don't pretend that a good collation exists. And don't try to solve all problems in one go
    PARAM_imroot="imroot";
    PARAM_kims="kims"; # only sample k images and crop from it to save hdd load time.
    OUTPUT_raw_images_np="raw_ims";
    PARAM_seed="seed";
    PARAM_buf_upd_prob="bufprob";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.masks_np = this.register_input(this.INPUT_masks_np, iocvt_dict);
        this.raw_images_np = this.register_output(this.OUTPUT_raw_images_np, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.imroot = neko_get_arg(this.PARAM_imroot, params);
        this.kims = neko_get_arg(this.PARAM_kims, params,3);
        this.all_bg=glob.glob(os.path.join(this.imroot,"*.jpg"));
        this.rng = random.Random(neko_get_arg(this.PARAM_seed,params,9));
        this.buffer_upd_prob=neko_get_arg(this.PARAM_buf_upd_prob,params,0.5);
        this.background_types=[0,1,2,3,3,3];
        this.bgbuffer=[];
        for i in range(this.kims):
            this.update_buffer(); # fill the buffer
        pass;
    def getimg(this):
        try:
            p=this.rng.choice(this.all_bg);
            i=cv2.imread(p);
            if(i is None):
                return this.getimg();
            return i;
        except:
            return this.getimg()
    def mix(this,masks_np,cs,bgt,bgi):
        return mix_with_gen_bg_serial(masks_np,cs,bgt,bgi);
    def rand_eye(this,img,roisz):
        y_sta=this.rng.randint(0,int(img.shape[0]*0.8));
        x_sta = this.rng.randint(0, int(img.shape[1] * 0.8));
        try:
            y_end=this.rng.randint(y_sta+1,min(y_sta+5*roisz[0],img.shape[0])); # stupidly large image and stupit
            x_end = this.rng.randint(x_sta+1,min(x_sta+5*roisz[1],img.shape[1]) );
            if(x_end-x_sta>5 and y_end-y_sta>5):
                return img[y_sta:y_end,x_sta:x_end];
            else:
                return img;
        except:
            return img;
    def update_buffer(this):
        if(len(this.bgbuffer)==this.kims):
            this.bgbuffer.pop(0);
        this.bgbuffer.append(cv2.cvtColor( this.getimg(),cv2.COLOR_BGR2RGB));

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.buffer_upd_prob> this.rng.random()):
            this.update_buffer();

        masks_np = workspace.get(this.masks_np);
        bgt=[ this.rng.choice(this.background_types) for m in masks_np];
        bgi=[];
        for id in range(len(bgt)):
            if(masks_np[id] is not None):
                t=bgt[id];
                if(t<3):
                    bgi.append(None);
                else:
                    bgi.append(this.rand_eye(this.rng.choice(this.bgbuffer),masks_np[id].shape))
            else:
                bgi.append(None);

        cs=[(this.rng.randint(0,255),this.rng.randint(0,255),this.rng.randint(0,255)) for m in masks_np];
        #
        am=this.mix(masks_np, cs, bgt, bgi);
        workspace.add(this.raw_images_np,am);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   masks_np,
                   raw_images_np,
                   imroot, kims
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_masks_np: masks_np, cls.OUTPUT_raw_images_np: raw_images_np},
                                         cls.PARAM_imroot: imroot, cls.PARAM_kims: kims, "modcvt_dict": {}}}
# class neko_random_bg_agent_parallel(neko_random_bg_agent):
#     def set_etc(this, params):
#         super().set_etc(params);
#         this.pool=Pool(neko_get_arg("workers",params,12));
#
#     def mix(this,masks_np,cs,bgt,bgi):
#         return mix_with_gen_bg_in_parallel(masks_np,cs,bgt,bgi,pool=this.pool);
# def get_neko_random_bg_agent(
#         masks_np,
#         raw_images_np,
#         imroot, kims
# ):
#     engine = neko_random_bg_agent;
#     return {"agent": engine,
#             "params": {"iocvt_dict": {engine.INPUT_masks_np: masks_np, engine.OUTPUT_raw_images_np: raw_images_np},
#                        engine.PARAM_imroot: imroot, engine.PARAM_kims: kims, "modcvt_dict": {}}}
#

if __name__ == '__main__':
    neko_random_bg_agent.print_default_setup_scripts()