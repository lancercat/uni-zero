import copy

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgr
import random

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
import os
# apply trdg mask generation for any text mask...
# mk1 does not rely on tdict. Bcs it just parrots input characters.
# mk2 will make tdict based augment, but that's mk2.

class neko_random_grounded_font_fetching_agent(neko_module_wrapping_agent):
    INPUT_text_label="text_label";

    OUTPUT_fnt_choices = "font_choices";
    OUTPUT_tfdict="tfdict";


    PARAM_fnt_idx_file="fnt_idx_file";
    PARAM_fnt_path="fnt_path";

    PARAM_rand_k="randk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.text_label = this.register_input(this.INPUT_text_label, iocvt_dict);
        this.fnt_choices = this.register_output(this.OUTPUT_fnt_choices, iocvt_dict);
        this.tfdict = this.register_output(this.OUTPUT_tfdict, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.fnt_idx_file = neko_get_arg(this.PARAM_fnt_idx_file, params);
        this.fnt_path = neko_get_arg(this.PARAM_fnt_path, params);
        this.rand_k = neko_get_arg(this.PARAM_rand_k, params, 3);
        this.fntmgr = fntmgr();
        this.fntmgr.tryload(this.fnt_idx_file, this.fnt_path,32);
        this.rng = random.Random(9);

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        text_label = workspace.get(this.text_label);
        afs=[];
        tfd = {};

        
        for tid in range(len(text_label)):
            fs=[os.path.join(this.fnt_path,f ) for f in this.fntmgr.fonts_available(text_label[tid])];

            sfs=this.rng.choices(fs,k=this.rand_k);
            afs.append(sfs);
            for f in sfs:
                if(f in tfd):
                    continue;
                # cs=this.fntmgr.get_cs(f);
                if(f not in tfd):
                    tfd[f]=set();
                tfd[f].union(set(tokenize(text_label[tid])));
        workspace.add(this.tfdict,tfd);
        workspace.add(this.fnt_choices,afs);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                    text_label,
                   fnt_choices, tfdict,
                   fnt_idx_file, fnt_path, rand_k
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": { cls.INPUT_text_label: text_label, cls.OUTPUT_fnt_choices: fnt_choices,
                           cls.OUTPUT_tfdict: tfdict}, cls.PARAM_fnt_idx_file: fnt_idx_file, cls.PARAM_fnt_path: fnt_path,
            cls.PARAM_rand_k: rand_k, "modcvt_dict": {}}}

def get_neko_random_grounded_font_fetching_agent(
        text_label,
        fnt_choices, tfdict,
        fnt_idx_file, fnt_path, rand_k
):
    engine = neko_random_grounded_font_fetching_agent;
    return {"agent": engine, "params": {
        "iocvt_dict": { engine.INPUT_text_label: text_label, engine.OUTPUT_fnt_choices: fnt_choices,
                       engine.OUTPUT_tfdict: tfdict}, engine.PARAM_fnt_idx_file: fnt_idx_file,
        engine.PARAM_fnt_path: fnt_path, engine.PARAM_rand_k: rand_k, "modcvt_dict": {}}}


# generates direct commands to the mask maker.
class neko_random_font_shuff(neko_module_wrapping_agent):
    INPUT_text_label="text_label";
    INPUT_fnt_choices="fnt_choices";
    INPUT_token_font_dict="tfdcit";

    OUTPUT_fnts="synth_fonts";
    OUTPUT_synth_text_label="synth_text_label";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.fnt_choices = this.register_input(this.INPUT_fnt_choices, iocvt_dict);
        this.text_label = this.register_input(this.INPUT_text_label, iocvt_dict);
        this.token_font_dict = this.register_input(this.INPUT_token_font_dict, iocvt_dict);
        this.fnts = this.register_output(this.OUTPUT_fnts, iocvt_dict);
        this.synth_text_label = this.register_output(this.OUTPUT_synth_text_label, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.rng=random.Random(9);
        pass;

    def alttext(this,t):
        tx = tokenize(t);
        this.rng.shuffle(tx)
        return "".join(tx);

    # simply permute each text and render it with
    # does not deconfund the character frequency but... I have only four paws.
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fnt_choices = workspace.get(this.fnt_choices);
        text_label = workspace.get(this.text_label);
        token_font_dict = workspace.get(this.token_font_dict);
        ft,ff=[],[];
        for t,f in zip(text_label,fnt_choices):
            tx=this.alttext(t);
            ft.append(tx);
            ff.append(f[0]);
        workspace.add(this.synth_text_label,ft);
        workspace.add(this.fnts, ff);
        return workspace, environment;


    @classmethod
    def get_agtcfg(cls,
                   fnt_choices, text_label, token_font_dict,
                   fnts, synth_text_label
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_fnt_choices: fnt_choices, cls.INPUT_text_label: text_label,
                           cls.INPUT_token_font_dict: token_font_dict, cls.OUTPUT_fnts: fnts,
                           cls.OUTPUT_synth_text_label: synth_text_label}, "modcvt_dict": {}}}
class neko_random_font_repeater(neko_random_font_shuff):
    def alttext(this,t):
        return t;



def get_neko_random_shuffle_text_one_font_gen(
        fnt_choices, text_label, token_font_dict,
        fnts, synth_text_label
):
    engine = neko_random_font_shuff;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_fnt_choices: fnt_choices, engine.INPUT_text_label: text_label,
                       engine.INPUT_token_font_dict: token_font_dict, engine.OUTPUT_fnts: fnts,
                       engine.OUTPUT_synth_text_label: synth_text_label}, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_random_font_shuff.print_default_setup_scripts();


