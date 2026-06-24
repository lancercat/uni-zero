from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.names import default_ocr_variable_names as dvn

import cv2
import numpy as np

class origin_fe_sub(neko_module_wrapping_agent):
    INPUT_image_name=dvn.raw_image_name;
    OUTPUT_feature_name = dvn.word_feature_name;
    MOD_feature_extractor_name="feature_extractor_name";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.image_name =  this.register(this.INPUT_image_name,iocvt_dict,this.input_dict);
        this.feature_name =  this.register(this.OUTPUT_feature_name,iocvt_dict,this.output_dict);
        this.fe_name =  this.register(this.MOD_feature_extractor_name,modcvt_dict,this.mnames);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        inp=workspace.inter_dict[this.image_name];
        # cv2.imshow("netin", (inp[0] * 127 + 127).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8));
        # cv2.imshow("meow", (coled[0] * 127 + 127).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8));
        # cv2.waitKey(0);
        features = environment.module_dict[this.fe_name](inp);
        features = [f.contiguous() for f in features];
        workspace.inter_dict[this.feature_name]=features;
        return workspace,environment;

# will pick its name from the execution plan


