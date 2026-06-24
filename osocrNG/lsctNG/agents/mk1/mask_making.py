import random
from multiprocess.pool import Pool

import numpy as np

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.lsctNG.libtrdg.mask_making_engine import neko_string_mask_generator,neko_string_mask_generator_insmask
from neko_sdk.cfgtool.argsparse import neko_get_arg
from PIL import Image
from neko_sdk.log import fatal,info,warn

class neko_mask_render_agent(neko_module_wrapping_agent):
    INPUT_text_labels="text_labels";
    INPUT_fonts = "fonts";

    OUTPUT_mask_im="mask_ims";
    OUTPUT_render_success="render_success";

    OUTPUT_mask_orient="mask_orient";
    PARAM_size="size";
    PARAM_orientation="orientation";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.fonts = this.register_input(this.INPUT_fonts, iocvt_dict);
        this.text_labels = this.register_input(this.INPUT_text_labels, iocvt_dict);
        this.mask_im = this.register_output(this.OUTPUT_mask_im, iocvt_dict);

        this.mask_orient = this.register_output(this.OUTPUT_mask_orient, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.orientation = neko_get_arg(this.PARAM_orientation, params);
        this.size = neko_get_arg(this.PARAM_size, params);
        this.mask_gen = neko_string_mask_generator({
            neko_string_mask_generator.PARAM_size: this.size,
            neko_string_mask_generator.PARAM_orientations: this.orientation,
        },random.Random(9));
        pass;
    def render(this,text_labels,fonts ):
        ms,oris=[],[];
        for t, f in zip(text_labels, fonts):
            m, o = this.mask_gen.try_drive(f, t);
            ms.append(m);
            oris.append(o);
        return ms, oris;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fonts = workspace.get(this.fonts);
        text_labels = workspace.get(this.text_labels);
        render_success=[];
        ms,oris=this.render(text_labels,fonts);
        for m in ms:
            if(m is None):
                render_success.append(False);
            else:
                render_success.append(True);

        workspace.add(this.mask_im,ms);
        workspace.add(this.mask_orient,oris);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   fonts, text_labels,
                   mask_im, mask_orient,
                   orientation, size
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_fonts: fonts, cls.INPUT_text_labels: text_labels,
                           cls.OUTPUT_mask_im: mask_im, cls.OUTPUT_mask_orient: mask_orient},
            cls.PARAM_orientation: orientation, cls.PARAM_size: size, "modcvt_dict": {}}}
class neko_mask_render_agent_parallel(neko_mask_render_agent):
    def set_etc(this, params):
        super().set_etc(params);
        this.pool=Pool(neko_get_arg("workers",params,12));
    def render(this,text_labels,fonts ):
        return this.mask_gen.try_drive_para(text_labels,fonts,pool=this.pool);


class neko_mask_render_agent_insmask(neko_module_wrapping_agent):
    INPUT_text_labels="text_labels";
    INPUT_fonts = "fonts";

    OUTPUT_mask_im="mask_ims";
    OUTPUT_mask_ins="mask_ins";
    OUTPUT_ins_utf="ins_utf";
    OUTPUT_mask_orient="mask_orient";
    PARAM_size="size";
    PARAM_orientation="orientation";
    PARAM_charspace="char_space";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.fonts = this.register_input(this.INPUT_fonts, iocvt_dict);
        this.text_labels = this.register_input(this.INPUT_text_labels, iocvt_dict);
        this.mask_im = this.register_output(this.OUTPUT_mask_im, iocvt_dict);
        this.mask_ins = this.register_output(this.OUTPUT_mask_ins, iocvt_dict);
        this.ins_utf=this.register_output(this.OUTPUT_ins_utf,iocvt_dict);
        this.mask_orient = this.register_output(this.OUTPUT_mask_orient, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.orientation = neko_get_arg(this.PARAM_orientation, params);
        this.size = neko_get_arg(this.PARAM_size, params);
        this.charspace=neko_get_arg(this.PARAM_charspace,params);

        this.mask_gen = neko_string_mask_generator_insmask({
            neko_string_mask_generator.PARAM_size: this.size,
            neko_string_mask_generator.PARAM_orientations: this.orientation,
            neko_string_mask_generator.PARAM_charspace:this.charspace
        },random.Random(9));
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fonts = workspace.get(this.fonts);
        text_labels = workspace.get(this.text_labels);
        ms,insms,oris,ts=[],[],[],[];
        for t, f in zip(text_labels,fonts):
            m,i,o=this.mask_gen.drive(f,t);
            t=this.mask_gen.tokenize(t); # now the instance label is synced with insmask
            ts.append(t);
            ms.append(m);
            insms.append(i);
            oris.append(o);
        workspace.add(this.ins_utf,ts); # instance utfs are now returned explicitly
        workspace.add(this.mask_im,ms);
        workspace.add(this.mask_ins,insms);
        workspace.add(this.mask_orient,oris);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   fonts, text_labels,
                   mask_im,mask_ins, mask_utf,mask_orient,
                   orientation, size,chaspace=None
                   ):
        if chaspace is None:
            chaspace=[0,32,64];
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_fonts: fonts, cls.INPUT_text_labels: text_labels,
                           cls.OUTPUT_mask_im: mask_im,cls.OUTPUT_mask_ins:mask_ins, cls.OUTPUT_ins_utf:mask_utf, cls.OUTPUT_mask_orient: mask_orient},
            cls.PARAM_orientation: orientation, cls.PARAM_size: size, cls.PARAM_charspace: chaspace, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_mask_render_agent.print_default_setup_scripts();
