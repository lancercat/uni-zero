import copy

import PIL.Image
import torch

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgr
import random
from neko_sdk.neko_framework_NG.data.supportdb.support_utf_im_lmdb import neko_random_support_fetcher
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
import numpy as np
# apply trdg mask generation for any text mask...

# this one generates its own meta.
class neko_random_lsct_char_glyphdb_headless_agent(neko_module_wrapping_agent):
    OUTPUT_text_label = "text_label";
    OUTPUT_text_masks="masks";
    OUTPUT_selected_utf = "selected_utf";

    PARAM_db_root="dbroot";
    PARAM_rand_k = "randk";  # generate k for each font/
    PARAM_seed="seed";
    PARAM_N="n_classes";
    PARAM_charset="charset";

    BIG_PRIME = 1000000009;

    pass;


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.text_label = this.register_output(this.OUTPUT_text_label, iocvt_dict);
        this.text_mask = this.register_output(this.OUTPUT_text_masks, iocvt_dict);

        this.selected_utf=neko_get_arg(this.OUTPUT_selected_utf,iocvt_dict);

    def set_etc(this, params):
        super().set_etc(params);
        seed=neko_get_arg(this.PARAM_seed,params,9);
        this.n=neko_get_arg(this.PARAM_N,params,512);
        this.k=neko_get_arg(this.PARAM_rand_k,params,2);
        this.rng = np.random.default_rng(seed);
        this.charset=neko_get_arg(this.PARAM_charset,params); # characterset has to be set explicitly...
        this.db_root=neko_get_arg(this.PARAM_db_root,params);
        this.db=neko_random_support_fetcher({
            neko_random_support_fetcher.PARAM_root_dict:
                {"main": this.db_root},
            neko_random_support_fetcher.PARAM_seed:seed,
        });

        assert len(this.charset)> this.n;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        selected_utf=this.rng.choice(this.charset,this.n,replace=False);
        selected_utf=[x for x in selected_utf if x];
        amsk=[];
        autf=[];
        for u in selected_utf:
            selected_msks = this.db.fetch(u,this.k);
            amsk.extend([PIL.Image.fromarray(selected_msk) for selected_msk in selected_msks]);
            autf.extend([u for _ in selected_msks]); # bcs it may fetch less than k for whatever reasons duh.
        workspace.add(this.selected_utf,selected_utf);
        workspace.add(this.text_label,autf);
        workspace.add(this.text_mask,amsk);
        return workspace,environment

    @classmethod
    def get_agtcfg(cls,
                   selected_utf, text_label, text_masks,
                   N, charset, db_root, rand_k, seed
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.OUTPUT_selected_utf: selected_utf, cls.OUTPUT_text_label: text_label,
                           cls.OUTPUT_text_masks: text_masks}, cls.PARAM_N: N, cls.PARAM_charset: charset,
            cls.PARAM_db_root: db_root, cls.PARAM_rand_k: rand_k, cls.PARAM_seed: seed, "modcvt_dict": {}}}

