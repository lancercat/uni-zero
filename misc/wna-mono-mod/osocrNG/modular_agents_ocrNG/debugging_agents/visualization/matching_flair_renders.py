from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import numpy as np
import editdistance
class neko_perfect_match_flair_making_agent(neko_module_wrapping_agent):
    INPUT_pred_text = "pred_text";
    INPUT_text_label = "text_label";
    OUTPUT_flairs= "all_flairs";
    INPUT_tdict = "tdict";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.pred_text = this.register_input(this.INPUT_pred_text, iocvt_dict);
        this.text_label = this.register_input(this.INPUT_text_label, iocvt_dict);
        this.all_flairs = this.register_output(this.OUTPUT_flairs, iocvt_dict);


        pass;



    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        pred_text = workspace.get(this.pred_text);
        try:
            text_label = workspace.get(this.text_label);
        except:
            text_label=None
        afs=[];
        for i in range(len(pred_text)):

            pred_flair = np.zeros([32, 32, 3], dtype=np.uint8);

            if(text_label is not None):
                feas = 1 - min(1, editdistance.eval(pred_text[i].lower(), text_label[i].lower()) / len(text_label[i]));
                if (text_label[i].lower() == pred_text[i].lower()):
                    pred_flair[:, :, 1] = 255;
                else:
                    pred_flair[:, :, -1] = 255;
            else:
                feas=1;
                pred_flair[:] = 255;
            qflair = (np.zeros([32, 32, 3]) + feas * 255)
            flairs = [pred_flair,qflair];
            afs.append(flairs);
        workspace.add(this.all_flairs,afs);
        return workspace, environment;

def get_neko_perfect_match_flair_making_agent(
        pred_text, text_label,
    all_flairs

):
    engine = neko_perfect_match_flair_making_agent;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_pred_text: pred_text, engine.INPUT_text_label: text_label,
                       engine.OUTPUT_flairs: all_flairs}, "modcvt_dict": {}}}

