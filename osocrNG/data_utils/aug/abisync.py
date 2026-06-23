# coding:utf-8

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import cv2
from third_eye.libabi.transforms import get_abi_aug_fixed
from PIL.Image import Transpose
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama


# abi aug does not support masks, for masked lsctw we will need to fallback to qhbaug, or no aug et al.
class neko_fixed_abinet_aug_agent(ama):
    PARAM_seed="NEP_not_impl_NEP"
    PARAM_BORDER_MOD="border_mod"
    INPUT_images_pre_aug="images_pre_aug";
    OUTPUT_images_post_aug="images_post_aug";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.images_post_aug = this.register_output(this.OUTPUT_images_post_aug, iocvt_dict);
        this.images_pre_aug = this.register_input(this.INPUT_images_pre_aug, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.bordermode = neko_get_arg(this.PARAM_BORDER_MOD, params, cv2.BORDER_REPLICATE);
        this.transform = get_abi_aug_fixed(this.bordermode);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        images_pre_aug = workspace.get(this.images_pre_aug);
        augged=this.process(images_pre_aug);
        workspace.add(this.images_post_aug,augged);
        return workspace, environment;


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
        return imgap;
    def process(this,images):
        # rngs=[None for _ in range(len(images))];
        l=[this.process_image(i) for i in list(images)]
        return l;
    @classmethod
    def get_agtcfg(cls,
        images_pre_aug,
        images_post_aug,
        border_mode,seed
    ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_images_pre_aug: images_pre_aug, cls.OUTPUT_images_post_aug: images_post_aug}, cls.PARAM_BORDER_MOD: border_mode, cls.PARAM_seed: seed, "modcvt_dict": {}}}


