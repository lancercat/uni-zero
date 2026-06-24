import copy

import torch

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgr
import random

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
import numpy as np
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama

# forks muxed meta and mask, remove non printable, and build tdict.
class lsctc_meta_fork(ama):
    INPUT_selected_utf="selected_utf";
    # INPUT_text_labels = "text_labels";
    INPUT_muxed_masks="muxed_masks";
    # INPUT_muxed_text_labels="muxed_text_labels";

    OUTPUT_tdict = "lsct_tdict";  # lsct will use its own tdict--- there is a good chance that we know the case.
    OUTPUT_proto_labels = "proto_labels";
    OUTPUT_proto_masks = "proto_masks";

    OUTPUT_sample_masks = "sample_masks";
    OUTPUT_sample_labels_utf = "sample_labels_utf";  # just generate them here to avoid the hassle---
    OUTPUT_sample_orient="sample_orient";
    PARAM_K="k";
    PARAM_sgmtsize="sgmtsize";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.masks = this.register_input(this.INPUT_muxed_masks, iocvt_dict);
        this.selected_utf = this.register_input(this.INPUT_selected_utf, iocvt_dict);
        this.proto_labels = this.register_output(this.OUTPUT_proto_labels, iocvt_dict);
        this.proto_masks = this.register_output(this.OUTPUT_proto_masks, iocvt_dict);

        this.sample_masks = this.register_output(this.OUTPUT_sample_masks, iocvt_dict);
        this.sample_labels = this.register_output(this.OUTPUT_sample_labels_utf, iocvt_dict);
        this.sample_orient=this.register_output(this.OUTPUT_sample_orient,iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_tdict, iocvt_dict);

        pass;

    def set_etc(this, params):
        this.K = neko_get_arg(this.PARAM_K, params);
        this.sgmtsize=neko_get_arg(this.PARAM_sgmtsize,params,this.K+1);
        pass;
    def arm(this,msk_segment,label):
        avm=[x for x in msk_segment if x is not None];
        if(len(avm)==0):
            return None,None,None;
        return [np.array(avm[0])],avm,[label for _ in avm];

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        masks = workspace.get(this.masks);
        selected_utf = workspace.get(this.selected_utf);
        td = {};
        all_samples=[];
        all_labels=[];
        all_protos = [];
        all_plabels=[];
        pid=0;
        for id in range(len(selected_utf)):
            sta=id*this.sgmtsize;
            chutf = str(selected_utf[id]);
            proto_batch, samples_batch,label_batch=this.arm(masks[sta:sta+this.sgmtsize],chutf);
            if(proto_batch is None):
                continue;  # if all characters are not printable, then forget about it
            td[pid] = chutf;
            td[chutf] = pid;
            pid += 1;
            all_protos+=proto_batch;
            all_labels += label_batch;
            all_samples+=samples_batch;
            all_plabels+=[chutf for _ in proto_batch];

        workspace.add(this.tdict,td); #
        workspace.add(this.sample_labels,all_labels);
        workspace.add(this.sample_masks,all_samples);
        workspace.add(this.proto_masks,all_protos);
        workspace.add(this.proto_labels,all_plabels);
        workspace.add(this.sample_orient,[0 for _ in all_samples]); # don't enforce orientation here, hell no.

        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   muxed_masks, selected_utf,
                   proto_labels, proto_masks, sample_labels_utf, sample_masks, sample_orient, tdict,
                   K
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_muxed_masks: muxed_masks, cls.INPUT_selected_utf: selected_utf,
                           cls.OUTPUT_proto_labels: proto_labels, cls.OUTPUT_proto_masks: proto_masks,
                           cls.OUTPUT_sample_labels_utf: sample_labels_utf, cls.OUTPUT_sample_masks: sample_masks,
                           cls.OUTPUT_sample_orient: sample_orient, cls.OUTPUT_tdict: tdict}, cls.PARAM_K: K,
            cls.PARAM_sgmtsize: K, "modcvt_dict": {}}}


if __name__ == '__main__':
    lsctc_meta_fork.print_default_setup_scripts()