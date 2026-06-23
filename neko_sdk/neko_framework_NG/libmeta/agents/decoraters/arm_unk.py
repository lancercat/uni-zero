

# decorates protos and tdict

import torch


from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.neko_framework_NG.names import default_variable_names as dvn
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
import copy

from osocrNG.sptokens import tUNKREP,tUNK,tDC,ID_DC
# why separation?
# well don't mess with different thing together.
# note label utf does not necessarily match proto utf
# however, if
class arm_centerless_to_meta(ama):
    INPUT_in_dict="tdict";
    INPUT_label_utf="label_utf";
    OUTPUT_label_utf_wunk = "label_utf_wunk";
    OUTPUT_tdict_wunk = "tdict_wunk";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.label_utf = this.register_input(this.INPUT_label_utf, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_in_dict, iocvt_dict);
        this.label_utf_wunk = this.register_output(this.OUTPUT_label_utf_wunk, iocvt_dict);
        this.tdict_wunk = this.register_output(this.OUTPUT_tdict_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;


    @classmethod
    def get_agtcfg(cls,
                   label_utf, tdict,
                   label_utf_wunk, tdict_wunk
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_label_utf: label_utf, cls.INPUT_in_dict: tdict,
                                                        cls.OUTPUT_label_utf_wunk: label_utf_wunk,
                                                        cls.OUTPUT_tdict_wunk: tdict_wunk}, "modcvt_dict": {}}}

# logit less --- it does not have a logit. Or rather, it only tells something to the loss function.
class arm_logitless_to_meta(ama):
    INPUT_in_dict="tdict";
    OUTPUT_tdict_wunk = "tdict_wunk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tdict = this.register_input(this.INPUT_in_dict, iocvt_dict);
        this.tdict_wunk = this.register_output(this.OUTPUT_tdict_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;


    @classmethod
    def get_agtcfg(cls, tdict,tdict_wunk):
        return {"agent": cls, "params": {"iocvt_dict": { cls.INPUT_in_dict: tdict,
                                                        cls.OUTPUT_tdict_wunk: tdict_wunk}, "modcvt_dict": {}}}


class arm_unknown_to_meta(arm_centerless_to_meta):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        label_utf = workspace.get(this.label_utf);
        tdict = workspace.get(this.tdict);
        ntdict = copy.copy(tdict);
        nutf = copy.copy(label_utf)
        ml = max(tdict, key=lambda k: k if isinstance(k, int) else -9)+1; # add the label as the last
        ntdict[ml] = tUNKREP;
        ntdict[tUNK] = ml;
        nutf.append(tUNK);
        workspace.add(this.tdict_wunk, ntdict);
        workspace.add(this.label_utf_wunk,nutf);
        return workspace, environment;

class arm_dont_care_to_meta(arm_logitless_to_meta):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        ntdict = copy.copy(tdict);
        ntdict[tDC] = ID_DC;
        workspace.add(this.tdict_wunk, ntdict);
        return workspace, environment;



class arm_unknown_to_confidence(ama):
    INPUT_confidence="confidence";
    OUTPUT_confidence_wunk="confidence_wunk";
    MOD_unk_threshold="unk_threshold";
    PARAM_dim="dim";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.confidence = this.register_input(this.INPUT_confidence, iocvt_dict);
        this.confidence_wunk = this.register_output(this.OUTPUT_confidence_wunk, iocvt_dict);
        this.unk_threshold = this.register_mod(this.MOD_unk_threshold, modcvt_dict);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        confidence = workspace.get(this.confidence);
        threshed_conf=this.unk_threshold(confidence); # adds threshold to whatever confidences
        workspace.add(this.confidence_wunk,threshed_conf);
        return workspace;


    @classmethod
    def get_agtcfg(cls,
                   confidence,
                   confidence_wunk,
                   unk_threshold,
                   dim
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_confidence: confidence,
                           cls.OUTPUT_confidence_wunk: confidence_wunk},
            "modcvt_dict": {cls.MOD_unk_threshold: unk_threshold},
            cls.PARAM_dim: dim}}
# its up to the module to figure out how reference is used.
class arm_referenced_unknown_to_confidence(arm_unknown_to_confidence):
    INPUT_reference="reference";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.reference = this.register_input(this.INPUT_reference, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        reference=workspace.get(this.reference);
        confidence = workspace.get(this.confidence);
        threshed_conf=this.unk_threshold(confidence,reference);
        # adds threshold to whatever confidences, w.r.t reference
        # see you want to scale the threshold by the mode of feature vector
        # which turns it into a cosine threshold without taxing numeric stability of the rest of the network.
        # said said, can this thing be static? like move other stuff around it.
        workspace.add(this.confidence_wunk,threshed_conf);
        return workspace;
    @classmethod
    def get_agtcfg(cls,
                   confidence, reference,
                   confidence_wunk,
                   unk_threshold,
                   dim
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_confidence: confidence, cls.INPUT_reference: reference,
                           cls.OUTPUT_confidence_wunk: confidence_wunk},
            "modcvt_dict": {cls.MOD_unk_threshold: unk_threshold}, cls.PARAM_dim: dim}}
class convert_unk_in_tokenized_utf_label(ama):
    INPUT_in_dict="tdict";
    INPUT_tokenized_utf="tokenized_utf";

    OUTPUT_tokenized_utf_wunk="tokenized_utf_wunk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tdict = this.register_input(this.INPUT_in_dict, iocvt_dict);
        this.tokenized_utf = this.register_input(this.INPUT_tokenized_utf, iocvt_dict);
        this.tokenized_utf_wunk = this.register_output(this.OUTPUT_tokenized_utf_wunk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        tokenized_utf = workspace.get(this.tokenized_utf);
        ss=[];
        for s in tokenized_utf:
            ts=[];
            for t in s:
               if(t in tdict):
                   ts.append(t);
               else:
                   ts.append(tUNKREP);
            ss.append(ts)
        workspace.add(this.tokenized_utf_wunk,ss);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tdict, tokenized_utf,
                   tokenized_utf_wunk
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_in_dict: tdict, cls.INPUT_tokenized_utf: tokenized_utf,
                                                        cls.OUTPUT_tokenized_utf_wunk: tokenized_utf_wunk},
                                         "modcvt_dict": {}}}


if __name__ == '__main__':
    convert_unk_in_tokenized_utf_label.print_default_setup_scripts()