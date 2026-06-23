import numpy as np


from neko_sdk.neko_framework_NG.agents.debugging.neko_abstract_debugging_agent import neko_abstract_debugging_agent

import cv2
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class neko_raw_img_saver_agent(neko_abstract_debugging_agent):
    INPUT_raw_image="raw_image";
    PARAM_export_path="";
    DFT_saveprfx = "raw_img";
    DFT_postfx = ".jpg";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.raw_im=this.register_input(this.INPUT_raw_image,iocvt_dict);
        this.keys=[this.raw_im];

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        fns,rds=this.get_items(workspace);
        for i in range(len(fns)):
            cv2.imwrite(fns[i],rds[i].get(this.raw_im));
        return workspace,environment;

# let's save gt and text to different file
class neko_text_saving_agent(neko_abstract_debugging_agent):
    INPUT_text="text";
    DFT_saveprfx = None; # will force you to give it a name.
    DFT_postfx = ".jpg";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict,modcvt_dict);
        this.text = this.register_input(this.INPUT_text, iocvt_dict);
        this.keys=[this.text];

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        fns, rds = this.get_items(workspace);
        for i in range(len(fns)):
            with open(fns[i], "w") as fp:
                fp.writelines(rds[i].get(this.text));
        return workspace,environment


def get_neko_text_saving_agent(
    text,uid,
    dstpath,saveprfx
):
    engine = neko_text_saving_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_text: text, engine.INPUT_uid: uid}, engine.PARAM_dstpath: dstpath, engine.PARAM_saveprfx: saveprfx, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_raw_img_saver_agent.print_default_setup_scripts()