# coding:utf-8
import random
from multiprocessing import get_context

import numpy as np
import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.augmentation.qhbaug import qhbwarp
from neko_sdk.ocr_modules.io.data_tiding import neko_aligned_left_top_padding_beacon_np
import cv2
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from third_eye.libabi.transforms import get_abi_aug,get_abi_aug_fixed
from PIL.Image import Transpose
class determinstic_augmenter_mk2:
    PARAM_seed="seed";
    @ classmethod
    def augment(cls,imgp,rng):
        img=np.array(imgp);
        if(imgp.height<imgp.width*1.1 and imgp.width<imgp.height*1.1):
            return img;
        if(imgp.height>imgp.width):
            return qhbwarp(img.transpose(1,0,2),10,rng=rng).transpose(1,0,2);
        else:
            return qhbwarp(img,10,rng=rng);
    @classmethod
    def process_image(cls,args):
        imgp,rng=args;
        imga=cls.augment(imgp, rng);
        return imga,imga.shape[:-1];
    @classmethod
    def process_image_no_aug(cls,args):
        imgp,rng=args;
        imga=imgp;
        return imga,imga.shape[:-1];

    def __init__(this,param):
        this.rng=random.Random(neko_get_arg(this.PARAM_seed,param,9));

    def process(this,images,thread_pool=None):
        seeds=[this.rng.randint(0,0xFFFFFFFFFFFCA71) for _ in range(len(images))];
        rngs=[random.Random(s) for s in seeds];
        # rngs=[None for _ in range(len(images))];
        if(thread_pool is None):
            l=[this.process_image(i) for i in list(zip(images, rngs))]
        else:
            l=list(thread_pool.map(determinstic_augmenter_mk2.process_image,
                                        list(zip(images,rngs)))
                   );
        # return ;
        return l;
class abiaug_fixed:
    PARAM_seed="NEP_not_impl_NEP"
    BORDER_MOD=cv2.BORDER_REPLICATE

    def __init__(this,param):
        this.transform=get_abi_aug_fixed(this.BORDER_MOD);

    def augment(this, imgp):
        if (imgp.height < imgp.width * 1.1 and imgp.width < imgp.height * 1.1):
            return imgp;
        if (imgp.height > imgp.width):
            return this.transform(imgp.transpose(Transpose.ROTATE_90)).transpose(Transpose.ROTATE_270);
        else:
            return this.transform(imgp);
    def process_image(this,args):
        imgp = args;
        imgap =this.augment(imgp);
        imga = np.array(imgap);
        return imga, imga.shape[:-1];
    def process(this,images,thread_pool=None):
        # rngs=[None for _ in range(len(images))];
        if(thread_pool is None):
            l=[this.process_image(i) for i in list(images)]
        else:
            l=list(thread_pool.map(this.process_image,list(images)));
        # return ;
        return l;

class abiaug_zp(abiaug_fixed):
    BORDER_MOD=cv2.BORDER_CONSTANT

class abiaug(abiaug_fixed):
    # for historical reasons, duh.
    def __init__(this,param):
        this.transform=get_abi_aug(this.BORDER_MOD);





class augment_agent(neko_abstract_async_agent):
    PARAM_seed="seed";
    PARAM_augparam="augmenter_para";
    PARAM_augmenter_workers="augmenter_workers";
    EXPORT_Q="export_queue";
    IMOPRT_Q="import_queue";
    @classmethod
    def augagt(cls):
        return determinstic_augmenter_mk2;
    def setup(this,param):
        ace=this.augagt();
        augpara=neko_get_arg(this.PARAM_augparam,param,{ace.PARAM_seed:9});
        this.augmenter_workers=neko_get_arg(this.PARAM_augmenter_workers,param,9);
        this.augmenter=ace(augpara);
        # this.augmenter=param["augmenter"];
        this.export_q=neko_get_arg(this.EXPORT_Q,param);
        this.import_q=neko_get_arg(this.IMOPRT_Q,param);
    def action_loop(this):
        if(this.augmenter_workers==0):
            thread_pool=None;
        else:
            thread_pool= get_context("spawn").Pool(this.augmenter_workers);
        while this.status==this.STATUS_running:
            data=this.environment.queue_dict[this.import_q].get();
            # print("fetching raw");
            il=[i[RN.IMAGE]  for i in data ];
            il=this.augmenter.process(il,thread_pool);
            ddict={
                RN.IMAGE:[],
                RN.PREAUG:[],
                RN.LABEL:[],
                RN.SIZE:[],
                RN.ANCHOR:[],
                RN.UID:[]
            }
            for i in range(len(il)):
                ddict[RN.IMAGE].append(il[i][0]);
                ddict[RN.PREAUG].append(data[i][RN.IMAGE]);
                ddict[RN.SIZE].append(il[i][1]);
                ddict[ RN.LABEL].append(data[i][RN.LABEL]);
                ddict[RN.ANCHOR].append(data[i][RN.ANCHOR]); # we keep it as anchor is kinda a human tag on the data. e.g., these are from gg glasses, and these are from hololens, etc etc.
                ddict[RN.UID].append(data[i][RN.UID]); # we now track the uid of a image (dataset, id, etc, will be routed)
            # print("putting_augged");
            this.environment.queue_dict[this.export_q].put(ddict);

class augment_agent_abinet(augment_agent):
    @classmethod
    def augagt(cls):
        return abiaug;
class augment_agent_abinet_fixed(augment_agent):
    @classmethod
    def augagt(cls):
        return abiaug_fixed;
class augment_agent_abinet_zeropadding(augment_agent):
    @classmethod
    def augagt(cls):
        return abiaug_zp;



from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
# just resizes the image to a fixed-size thumbnail.
class neko_basic_beacon_agent(neko_module_wrapping_agent):
    PARAM_beacon_w = "beacon_w";
    PARAM_beacon_h = "beacon_h";
    INPUT_raw_im="image";
    OUTPUT_beacon="beacon";
    PARAM_mode="mode";

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.i=this.register_input(this.INPUT_raw_im,iocvt_dict);
        this.o=this.register_output(this.OUTPUT_beacon,iocvt_dict);
    def set_etc(this,param):
        this.size=(neko_get_arg(this.PARAM_beacon_w,param),neko_get_arg(this.PARAM_beacon_h,param));
        this.mode=neko_get_arg(this.PARAM_mode,param,cv2.INTER_AREA);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        workspace.add(this.o,[cv2.resize(i,this.size,interpolation=this.mode) for i in workspace.get(this.i)]);
        return workspace,environment;

def get_neko_basic_beacon_agent(raw_im_name,
beacon_name,
beacon_h,beacon_w,mode=cv2.INTER_AREA):
    engine = neko_basic_beacon_agent;
    return {"agent": engine,
    "params": {
        "iocvt_dict": {engine.INPUT_raw_im: raw_im_name, engine.OUTPUT_beacon: beacon_name},
        "modcvt_dict": {},
        engine.PARAM_beacon_h: beacon_h, engine.PARAM_beacon_w: beacon_w, engine.PARAM_mode: mode
    }};

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
# just resizes the image to a fixed-size thumbnail.
class neko_grid_beacon_agent(neko_module_wrapping_agent):
    PARAM_beacon_w = "beacon_w";
    PARAM_beacon_h = "beacon_h";
    PARAM_wpart="wpart";
    PARAM_hpart = "hpart";

    INPUT_raw_im="image";
    OUTPUT_beacon="beacon";
    PARAM_mode="mode";

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.i=this.register_input(this.INPUT_raw_im,iocvt_dict);
        this.o=this.register_output(this.OUTPUT_beacon,iocvt_dict);
    def set_etc(this,param):
        this.bw=neko_get_arg(this.PARAM_beacon_w,param);
        this.bh=neko_get_arg(this.PARAM_beacon_h,param);
        this.xw=neko_get_arg(this.PARAM_wpart,param);
        this.xh=neko_get_arg(this.PARAM_hpart,param);
        this.size=(this.bw,this.bh);
        this.lsize=(this.bw*this.xw,this.bh*this.xh);
        this.mode=neko_get_arg(this.PARAM_mode,param,cv2.INTER_AREA);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        t=[np.concatenate([
            cv2.resize(i, this.size, interpolation=this.mode),
            cv2.resize(i, this.lsize, interpolation=this.mode).reshape(
                [this.xh, this.bh, this.xw, this.bw, -1]).transpose(1, 3, 0, 2, 4).reshape([this.bh, this.bw, -1])],
            axis=-1) for i in workspace.get(this.i)];

        workspace.add(this.o,t);
        return workspace,environment;
class neko_gray_grid_beacon_agent(neko_grid_beacon_agent):
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        t=[np.concatenate([
            cv2.resize(i, this.size, interpolation=this.mode),
            cv2.resize(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY), this.lsize, interpolation=this.mode).reshape(
                [this.xh, this.bh, this.xw, this.bw, -1]).transpose(1, 3, 0, 2, 4).reshape([this.bh, this.bw, -1])],
            axis=-1) for i in workspace.get(this.i)];
        workspace.add(this.o,t);
        return workspace,environment;
def get_neko_grid_beacon_agent(raw_im_name,
beacon_name,
beacon_h,beacon_w,h_part,w_part,mode=cv2.INTER_AREA):
    engine = neko_grid_beacon_agent;
    return {"agent": engine,
    "params": {
        "iocvt_dict": {engine.INPUT_raw_im: raw_im_name, engine.OUTPUT_beacon: beacon_name},
        "modcvt_dict": {},
        engine.PARAM_beacon_h: beacon_h,
        engine.PARAM_beacon_w: beacon_w,
        engine.PARAM_hpart:h_part,
        engine.PARAM_wpart: w_part,
        engine.PARAM_mode: mode
    }};
def get_neko_gray_grid_beacon_agent(raw_im_name,
beacon_name,
beacon_h,beacon_w,h_part,w_part,mode=cv2.INTER_AREA):
    engine = neko_gray_grid_beacon_agent;
    return {"agent": engine,
            "params": {
                "iocvt_dict": {engine.INPUT_raw_im: raw_im_name, engine.OUTPUT_beacon: beacon_name},
                "modcvt_dict": {},
                engine.PARAM_beacon_h: beacon_h,
                engine.PARAM_beacon_w: beacon_w,
                engine.PARAM_hpart: h_part,
                engine.PARAM_wpart: w_part,
                engine.PARAM_mode: mode
            }};
class regulate_images(neko_module_wrapping_agent):
    INPUT_images="in_images";
    OUTPUT_images="out_images";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.ins=this.register(this.INPUT_images,iocvt_dict,{});
        for i in this.ins:
            this.input_dict[this.INPUT_images+"_"+str(i)]
        this.outs=this.register_input(this.OUTPUT_images,iocvt_dict,{});


if __name__ == '__main__':
    print(neko_basic_beacon_agent.get_default_configuration_scripts());
