from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.names import default_ocr_variable_names as dvn

import cv2
import numpy as np


class imfe_NG_sub(neko_module_wrapping_agent):
    INPUT_image_name="raw_images";
    OUTPUT_last_feature_name = "last_feature";
    OUTPUT_tower_name = "feature_tower";
    OUTPUT_tower_tag_name = "feature_tower_tags";

    MOD_feature_extractor_name="feature_extractor_name";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.images = this.register_input(this.INPUT_image_name, iocvt_dict);
        this.last_feature_name = this.register_output(this.OUTPUT_last_feature_name, iocvt_dict);
        this.tower_name = this.register_output(this.OUTPUT_tower_name, iocvt_dict);
        this.tower_tag_name = this.register_output(this.OUTPUT_tower_tag_name, iocvt_dict);
        this.feature_extractor_name = this.register_mod(this.MOD_feature_extractor_name, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        images = workspace.get(this.images);
        feature, tower, towertag = environment.module_dict[this.feature_extractor_name](images);
        workspace.add(this.last_feature_name,feature); # if contigous is needed call it in the mod or just use another agent in utils
        if(this.tower_name is not None):
            workspace.add(this.tower_name,tower);
            workspace.add(this.tower_tag_name,towertag);
        return workspace, environment;


    @classmethod
    def get_agtcfg(cls,
                   images,
                   last_feature_name, tower_name, tower_tag_name,
                   feature_extractor_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_image_name: images, cls.OUTPUT_last_feature_name: last_feature_name,
                           cls.OUTPUT_tower_name: tower_name, cls.OUTPUT_tower_tag_name: tower_tag_name},
            "modcvt_dict": {cls.MOD_feature_extractor_name: feature_extractor_name}}}




# will pick its name from the execution plan
if __name__ == '__main__':
    imfe_NG_sub.print_default_setup_scripts()

