import random

from neko_sdk.cfgtool.argsparse import neko_get_arg
import numpy as np


from osocrNG.lsctNG.libtrdg.trdg_core import generate
from PIL import Image
from neko_sdk.log import fatal,info,warn
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

def drive_stub(params):
    content, font, size, orientation, space_width, character_spacing, fit, word_split = params;
    raw_mask, insmasks = generate(
        content,
        font,
        size, orientation, space_width, character_spacing, fit, word_split
    );
    return raw_mask, orientation;

def try_drive_stub(params):
        try:
            m, o = drive_stub(params)
        except:
            return None, None
        if (m.size[0]<2 or m.size[1] < 2):
            m, o = None, None;
        elif (np.array(m).max() < 60):
            m, o = None, None
        return m,o;


def force_drive_stub(params):
    m, o = drive_stub(params)
    if (m.size[0] < 2 or m.size[1] < 2):
        m, o = None, None;
    elif (np.array(m).max() < 60):
        m, o = None, None
    return m, o;


class neko_string_mask_generator:
    PARAM_size = "size";
    PARAM_orientations="orientations";
    PARAM_charspace="charspace";
    def set_genpara(this,param):
        # generator
        # hyper parameter
        this.size = 64;
        this.word_split = [True, False];
        this.character_spacing =neko_get_arg(this.PARAM_charspace,param, [0, 64, 32]);
        this.fit = 0;
        this.orientation = neko_get_arg(this.PARAM_orientations,param,[0]);
        this.space_width = 1;
        this.size=neko_get_arg(this.PARAM_size,param,64);

    ### well, rng is not a configurable thing...
    def __init__(this,param,rng=None):
        # meta info
        # background image list
        this.set_genpara(param);
        if(rng is not None):
            this.rng=rng
        else:
            this.rng=random.Random();
        pass;

    def drive_param(this):
        size = this.size;
        orientation = this.rng.choice(this.orientation);
        space_width = this.space_width;
        character_spacing = this.rng.choice(this.character_spacing);
        fit = this.fit;
        word_split = False;
        return size,orientation,space_width,character_spacing,fit,word_split;






    # Drive the vehicle
    def drive(this,font,content):
        param=this.drive_param();
        return drive_stub((content,font,*param));
    def try_drive_para(this,fonts,contents,pool):
        aps=[(c,f,*this.drive_param()) for f,c in zip(fonts,contents)];
        mos=pool.map(try_drive_stub,aps);
        ms,os=[],[];
        for i in range(len(mos)):
            ms.append(mos[i][0]);
            os.append(mos[i][1]);
        return ms, os;
    def drive_dbg(this,fonts,contents,pool):
        mos=[force_drive_stub([c,f,*this.drive_param()]) for f,c in zip(fonts,contents)];
        ms, os = [], [];
        for i in range(len(mos)):
            ms.append(mos[i][0]);
            os.append(mos[i][1]);
        return ms,os

    def try_drive(this,font,content):
        param=this.drive_param();
        return try_drive_stub((content,font,*param));


class neko_string_mask_generator_insmask(neko_string_mask_generator):

    def set_genpara(this,param):
        # generator
        # hyper parameter
        this.size = 64;
        this.word_split = [True, False];
        this.character_spacing = neko_get_arg(this.PARAM_charspace,param,[0, 64, 32]);
        this.fit = 0;
        this.orientation = neko_get_arg(this.PARAM_orientations,param,[0]);
        this.space_width = 1;
        this.size=neko_get_arg(this.PARAM_size,param,64);

    ### well, rng is not a configurable thing...
    def __init__(this,param,rng=None):
        # meta info
        # background image list
        this.set_genpara(param);
        if(rng is not None):
            this.rng=rng
        else:
            this.rng=random.Random();
        pass;
    def tokenize(this,text):
        return tokenize(text);
    def empty(this):
        m = Image.new('RGB', (32, 32), (255, 255, 255));
        i = Image.new('RGB', (32, 32), (255, 255, 255));
        o=0;
        warn("what the bloody?");
        # https://github.com/python-pillow/Pillow/issues/5592
        return m, [i], o;
    def drive_unsafe(this,font,content,orientation=None):
        size = this.size;
        if (orientation is None):
            if (len(content) == 1):
                orientation = 0;
            else:
                orientation = this.rng.choice(this.orientation);

        space_width = this.space_width;
        character_spacing = this.rng.choice(this.character_spacing);
        fit = this.fit;
        word_split = False;
        raw_mask, insmasks = generate(
            content,
            font,
            size,
            orientation, space_width, character_spacing, fit, word_split
        );
        return raw_mask,insmasks,orientation;

    # Drive the vehicle
    def drive(this,font,content,orientation=None):
        try:
            raw_mask,insmasks,orientation=this.drive_unsafe(font,content,orientation);
        except:
            return this.empty();
        if (raw_mask.size[0] < 1):
            return this.empty();

        return raw_mask,insmasks,orientation;
