import json
import os.path

import cv2

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.log import fatal,info,warn
from osocrNG.sptokens import tUNK, tDC, tSPLIT_OCR
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

class ocr_simple_logging_agent(neko_module_wrapping_agent):
    INPUT_images="images";
    INPUT_pred="preds";
    INPUT_gt="gts";
    INPUT_global_uid="global_uid";
    PARAM_root="root";
    PARAM_uid_key="uid_key";
    DFT_uid_key=["id"];

    PARAM_prfx = "prefix";
    DFT_prfx="NEP_skipped_NEP"

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.global_uid = this.register_input(this.INPUT_global_uid, iocvt_dict);
        this.gt = this.register_input(this.INPUT_gt, iocvt_dict);
        this.images = this.register_input_list(this.INPUT_images, iocvt_dict);
        this.pred = this.register_input(this.INPUT_pred, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.root = neko_get_arg(this.PARAM_root, params);
        this.uid_key = neko_get_arg(this.PARAM_uid_key, params, this.DFT_uid_key);
        this.prfx = neko_get_arg(this.PARAM_prfx, params, this.DFT_prfx);

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.global_uid not in workspace.inter_dict):
            return workspace, environment;
        global_uid = workspace.get(this.global_uid);
        if("benchmark_name" in workspace.inter_dict):
            root=os.path.join(this.root,workspace.get("benchmark_name"));
        else:
            root=this.root;

        images = [(image,workspace.get(image)) for image in this.images];
        pred = workspace.get(this.pred);
        if(this.gt in workspace.inter_dict):
            try:
                gt = workspace.get(this.gt);
            except:
                gt =None;
        else:
            gt=None
        os.makedirs(root,exist_ok=True);
        for i in range(len(global_uid)):
            n = "-".join([str(global_uid[i][k]) for k in this.uid_key]); # well if the data has tags of different aspects.
            if(this.prfx is None):
                if(gt is not None):
                    with open(os.path.join(root,n+"-gt.txt"),"w+") as fp:
                        fp.writelines([gt[i]]);
                with open(os.path.join(root,n + "-pr.txt"), "w+") as fp:
                    fp.writelines([pred[i]]);
                for j in range(len(images)):
                    cv2.imwrite(os.path.join(root,n+"-"+images[j][0]+".png"),
                                images[j][1][i]);
            else:
                if(gt is not None):
                    with open(os.path.join(root, n + "-"+this.prfx+"-gt.txt"), "w+") as fp:
                        fp.writelines([gt[i]]);
                with open(os.path.join(root, n + "-"+this.prfx+ "-pr.txt"), "w+") as fp:
                    fp.writelines([pred[i]]);
                for j in range(len(images)):
                    cv2.imwrite(os.path.join(root, n + "-"+this.prfx+ "-" + images[j][0] + ".png"),
                                images[j][1][i]);
        return workspace, environment;
class ocr_beam_logging_agent(neko_module_wrapping_agent):
    INPUT_beam="beam";
    INPUT_global_uid="global_uid";
    INPUT_tdict="tdict";
    PARAM_root="root";
    PARAM_uid_key="uid_key";
    DFT_uid_key=["id"];

    PARAM_prfx = "prefix";
    DFT_prfx="NEP_skipped_NEP"

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.global_uid = this.register_input(this.INPUT_global_uid, iocvt_dict);
        this.beam = this.register_input_list(this.INPUT_beam, iocvt_dict);
        this.tdict=this.register_input(this.INPUT_tdict,iocvt_dict);
        pass;

    def set_etc(this, params):
        this.root = neko_get_arg(this.PARAM_root, params);

        this.uid_key = neko_get_arg(this.PARAM_uid_key, params, this.DFT_uid_key);
        this.prfx = neko_get_arg(this.PARAM_prfx, params, this.DFT_prfx);

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.global_uid not in workspace.inter_dict):
            return workspace, environment;
        if(this.beam not in workspace.inter_dict):
            warn("cannot find beams in the results, skipping")
        global_uid = workspace.get(this.global_uid);
        if("benchmark_name" in workspace.inter_dict):
            root=os.path.join(this.root,workspace.get("benchmark_name"));
        else:
            root=this.root;
        beam_utf = workspace.get(this.beam);
        tdict=workspace.get(this.tdict);


        os.makedirs(root,exist_ok=True);

        for i in range(len(global_uid)):
            beam = [];

            for ub in beam_utf[i]:
                beam.append([tdict[_] if _ in tdict else tdict[tUNK] for _ in tokenize(ub)]);
            n = "-".join([str(global_uid[i][k]) for k in this.uid_key]); # well if the data has tags of different aspects.
            if(this.prfx is None):
                with open(os.path.join(root,n + "-beam.json"), "w+") as fp:
                    json.dump({
                        "bad_image":0,
                         "gt": [],
                        "beam": beam,
                        "beam_utf": beam_utf[i],
                        "gt_length": 0
                    },fp);

            else:
                with open(os.path.join(root, n + "-"+this.prfx+ "-json.txt"), "w+") as fp:
                    json.dump({
                        "bad_image": 0,
                        "gt": [],
                        "beam": beam,
                        "beam_utf": beam_utf[i],
                        "gt_length": 0
                    }, fp);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tdict, beam, global_uid,
                   prfx, root, uid_key
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_tdict: tdict,cls.INPUT_beam: beam, cls.INPUT_global_uid: global_uid}, cls.PARAM_prfx: prfx,
                           cls.PARAM_root: root, cls.PARAM_uid_key: uid_key, "modcvt_dict": {}}}


def get_ocr_simple_logging_agent(
            global_uid, gt, images, pred,
        root, uid_key,prfx=None
    ):
    engine = ocr_simple_logging_agent;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_global_uid: global_uid, engine.INPUT_gt: gt, engine.INPUT_images: images,
                       engine.INPUT_pred: pred}, engine.PARAM_root: root, engine.PARAM_uid_key: uid_key,engine.PARAM_prfx:prfx,
        "modcvt_dict": {}}}


class ocr_gathered_logging_agent(neko_module_wrapping_agent):
    INPUT_all_ids="all_ids";
    PARAM_branches="branches";
    PARAM_log_root="log_root";
    INPUT_branch_ids="branch_ids";
    INPUT_branch_preds="branch_preds";
    INPUT_branch_gts = "branch_gts";
    INPUT_branch_figures="branch_figures"

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.all_ids = this.register_input(this.INPUT_all_ids, iocvt_dict);
        this.branch_figures = this.register_input_list(this.INPUT_branch_figures, iocvt_dict);
        this.branch_gts = this.register_input_list(this.INPUT_branch_gts, iocvt_dict);
        this.branch_ids = this.register_input_list(this.INPUT_branch_ids, iocvt_dict);
        this.branch_preds = this.register_input_list(this.INPUT_branch_preds, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.branches = neko_get_arg(this.PARAM_branches, params);
        this.log_root = neko_get_arg(this.PARAM_log_root, params);


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        all_ids = workspace.get(this.all_ids);
        ars={};

        for bid in range(len(this.branches)):
            if(this.branch_ids[bid] in workspace):
                abid=workspace.get(this.branch_ids[bid]);
                abpred=workspace.get(this.branch_preds[bid]);
                abfig=workspace.get(this.branch_figures[bid]);
                abgt=workspace.get(this.branch_figures[bid]);
                for i in range(len(abid)):
                    if(abid[i] not in ars):
                        ars[abid]={
                            "pred": [],
                            "gt": [],
                            "figure":[],
                        };
                    ars[abid]["pred"].append(abpred[i]);
                    ars[abid]["gt"].append(abgt[i]);
                    ars[abid]["figure"].append(abfig[i]);
        for k in ars:
            fn=os.path.join(this.log_root,k)
            with open(fn+"_gt.txt","w") as fp:
                fp.writelines([ars[abid]["gt"][0]]);
            with open(fn + "_pred.txt", "w") as fp:
                fp.writelines([ars[abid]["gt"][0]]);

        return workspace, environment;




class simple_log_saver_agent(neko_module_wrapping_agent):
    PARAM_save_location="save_location";
    INPUT_sam_id="samid";
    def set_etc(this, params):
        this.save_location = neko_get_arg(this.PARAM_save_location, params);
        os.path.join(this.save_location)
        pass;
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.sam_id=this.register_input(this.INPUT_sam_id,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        sam_id = workspace.get(this.sam_id);
        for i in range(len(sam_id)):
            ir=os.path.join(this.save_location,sam_id[i]);
            os.makedirs(ir,exist_ok=True);
            for k in workspace.logdict["image"]:
                cv2.imwrite(os.path.join(ir,k+".jpg"),workspace.logdict[k][i]);
            for t in workspace.logdict["texts"]:
                with open(os.path.join(ir,t+".txt"),"w+") as fp:
                    fp.writelines(workspace.logdict[t][i]);
        return workspace, environment;
def get_ocr_gathering_agent(
	all_ids,branch_ids,branch_figures,branch_gts,branch_preds,branches
):
	engine = ocr_gathered_logging_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_all_ids: all_ids, engine.INPUT_branch_figures: branch_figures, engine.INPUT_branch_gts: branch_gts, engine.INPUT_branch_ids: branch_ids, engine.INPUT_branch_preds: branch_preds}, engine.PARAM_branches: branches, "modcvt_dict": {}}}
def get_simple_log_saver_agent(
	sam_id,
	save_location
):
	engine = simple_log_saver_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_sam_id: sam_id}, engine.PARAM_save_location: save_location, "modcvt_dict": {}}}

if __name__ == '__main__':
    ocr_beam_logging_agent.print_default_setup_scripts();
