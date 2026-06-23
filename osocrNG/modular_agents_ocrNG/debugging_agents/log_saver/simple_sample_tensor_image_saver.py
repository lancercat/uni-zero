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
from osocrNG.modular_agents_ocrNG.debugging_agents.log_saver.simple_sample_image_saver import neko_virtual_logging_agent_mk2

class neko_simple_tensor_img_logging_agent(neko_virtual_logging_agent_mk2):
    INPUT_images="images";
    PARAM_mean="mean";
    PARAM_var = "var";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.images = this.register_input_list(this.INPUT_images, iocvt_dict);
        pass;

    def set_etc(this, params):
        super().set_etc(params);
        this.mean=neko_get_arg(this.PARAM_mean,params,127);
        this.var = neko_get_arg(this.PARAM_mean, params, 127);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        global_ids=workspace.get(this.global_uid)
        root=this.logdir(workspace);
        images = [(image,workspace.get(image).detach().cpu().permute(0,2,3,1)*this.var+this.mean) for image in this.images];
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
                   prfx, root, uid_key,
                   mean=127,var=127
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_global_uid: global_uid, cls.INPUT_images: images},
                                         cls.PARAM_prfx: prfx, cls.PARAM_root: root, cls.PARAM_uid_key: uid_key,
                                         cls.PARAM_mean: mean, cls.PARAM_var:var,
                                         "modcvt_dict": {}}}


