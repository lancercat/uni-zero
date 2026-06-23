import random;

import torch;

from neko_sdk.ocr_modules.renderlite.lib_render import render_lite;


class metafier:
    def __init__(this,tr=0,te=0,val=1):
        this.render=render_lite(os=84,fos=32);
        this.tr=tr;
        this.te=te;
        this.val=val;
        pass;
    def metafy(this,characters,sp_tokens,fonts,font_ids,proto_dst,split_dst,debug=True):
        this.render.render(characters,sp_tokens,fonts,font_ids,proto_dst,debug);
        spc=this.render.spaces;
        random.shuffle(characters);
        portion=len(characters)/(this.te+this.tr+this.val);
        this.tr_at=int(this.tr*portion);
        this.val_at = int(this.val * portion)+this.tr_at;
        tr_labels=characters[:this.tr_at];
        val_labels=characters[this.tr_at:this.val_at];
        test_labels=characters[this.val_at:];
        # Whatever unicode it is, white spaces are white spaces.
        torch.save({"spc":spc,"tr":tr_labels,"val":val_labels,"te":test_labels},split_dst);


