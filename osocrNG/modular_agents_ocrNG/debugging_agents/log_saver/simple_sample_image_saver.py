import json
import os.path

import cv2
import numpy as np

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.log import fatal,info,warn
from osocrNG.sptokens import tUNK, tDC, tSPLIT_OCR
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
class neko_virtual_logging_agent_mk2(neko_module_wrapping_agent):
    PARAM_root="root";
    INPUT_global_uid="global_uid";
    PARAM_uid_key="uid_key";
    DFT_uid_key=["id"];

    PARAM_prfx = "prefix";
    DFT_prfx="NEP_skipped_NEP"
    def logdir(this,workspace):
        if("benchmark_name" in workspace.inter_dict):
            root=os.path.join(this.root,workspace.get("benchmark_name"));
        else:
            root=this.root;
        return root;
    def set_etc(this, params):
        this.root = neko_get_arg(this.PARAM_root, params);
        this.prfx = neko_get_arg(this.PARAM_prfx, params, this.DFT_prfx);
        this.uid_key = neko_get_arg(this.PARAM_uid_key, params, this.DFT_uid_key);

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.global_uid = this.register_input(this.INPUT_global_uid, iocvt_dict);

class neko_simple_img_logging_agent(neko_virtual_logging_agent_mk2):
    INPUT_images="images";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.images = this.register_input_list(this.INPUT_images, iocvt_dict);
        pass;


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        global_ids=workspace.get(this.global_uid)
        root=this.logdir(workspace);
        images = [(image,workspace.get(image)) for image in this.images];
        os.makedirs(root,exist_ok=True);
        for i in range(len(global_ids)):
            n = "-".join([str(global_ids[i][k]) for k in this.uid_key]); # well if the data has tags of different aspects.
            for j in range(len(images)):
                cv2.imwrite(os.path.join(root, n + "-"+this.prfx+ "-" + images[j][0] + ".png"),
                            np.array(images[j][1][i]));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   global_uid, images,
                   prfx, root, uid_key
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_global_uid: global_uid, cls.INPUT_images: images},
                                         cls.PARAM_prfx: prfx, cls.PARAM_root: root, cls.PARAM_uid_key: uid_key,
                                         "modcvt_dict": {}}}



class neko_simple_seq_logging_agent(neko_virtual_logging_agent_mk2):
    INPUT_seqs="seqs";
    PARAM_flair="flare"; # gt/pred or whatever.
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.seqs= this.register_input_list(this.INPUT_seqs, iocvt_dict);
        pass;
    def set_etc(this, params):
        super().set_etc(params);
        this.flare=neko_get_arg(this.PARAM_flair,params);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        global_ids=workspace.get(this.global_uid)
        root=this.logdir(workspace);
        seqs = workspace.get(this.seqs);
        os.makedirs(root,exist_ok=True);
        for i in range(len(global_ids)):
            n = "-".join([str(global_ids[i][k]) for k in this.uid_key]); # well if the data has tags of different aspects.
            if(n=="1319"):
                pass;
            with open(os.path.join(root, n + "-" + this.prfx + "-"+this.flare+".txt"), "w+") as fp:
                fp.writelines([str(s) for s in seqs[i]]);

        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   global_uid, seqs,
                   flair, prfx, root, uid_key
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_global_uid: global_uid, cls.INPUT_seqs: seqs},
                                         cls.PARAM_flair: flair, cls.PARAM_prfx: prfx, cls.PARAM_root: root,
                                         cls.PARAM_uid_key: uid_key, "modcvt_dict": {}}}


if __name__ == '__main__':
    ocr_simple_img_logging_agent.print_default_setup_scripts();
