import cv2

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# some op does not assume multi-part feature. use this to unpart
# pads the input image by a default 5 percent, minimum 3 pixels
# how much the collate agent takes from the padding is another matter.

class neko_cv2_padding_agent(neko_module_wrapping_agent):
    INPUT_in_image="input";
    OUTPUT_out_image="out";
    PARAM_padding_frac="padding_frac";
    PARAM_min_padding_size="min_padding_size";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.in_image = this.register_input(this.INPUT_in_image, iocvt_dict);
        this.out_image = this.register_output(this.OUTPUT_out_image, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.min_padding_size = neko_get_arg(this.PARAM_min_padding_size, params);
        this.padding_frac = neko_get_arg(this.PARAM_padding_frac, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        in_image = workspace.get(this.in_image);
        out_image=[];
        for im in in_image:
            h,w=im.shape[:2];
            padh=int(max(h*this.padding_frac,this.min_padding_size));
            padw=int(max(w*this.padding_frac,this.min_padding_size));
            out_image.append(cv2.copyMakeBorder(im,padh,padh,padw,padw,cv2.BORDER_CONSTANT,None,(127,127,127)));
        workspace.add(this.out_image,out_image);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   in_image,
                   out_image,
                   min_padding_size, padding_frac
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_in_image: in_image, cls.OUTPUT_out_image: out_image},
                                         cls.PARAM_min_padding_size: min_padding_size,
                                         cls.PARAM_padding_frac: padding_frac, "modcvt_dict": {}}}

def get_neko_cv2_padding_agent(
        in_image,
        out_image,
        min_padding_size, padding_frac
):
    engine = neko_cv2_padding_agent;
    return {"agent": engine,
            "params": {"iocvt_dict": {engine.INPUT_in_image: in_image, engine.OUTPUT_out_image: out_image},
                       engine.PARAM_min_padding_size: min_padding_size,
                       engine.PARAM_padding_frac: padding_frac, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_cv2_padding_agent.print_default_setup_scripts();
