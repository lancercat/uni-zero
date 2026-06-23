import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNK

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
# well no sp tokens added, no length tensor added in this one.
# this will assume the thing being flattened
# if you want voodoo, make new agents

# in NG2 this is tokenized.
# The datasource decides how to tokenize,
# if you want some ngram-based shit,
# retokenize it and make new tdict and plabel,
# arm new corrsponding protos aswell
# also don't use this if you are up to some multi-hot labels
# this thing can be used for simple single labeled object classification, so it sits here instead of OSOCRNG
class neko_abstract_label_maker(ama):
    INPUT_sam_utf="sam_utf"; # list of list of tokens.
    INPUT_tdict = "tdict";
    INPUT_dev_indicator = "dev_indictor";
    pass;
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.dev_indicator = this.register_input(this.INPUT_dev_indicator, iocvt_dict);
        this.sam_utf = this.register_input(this.INPUT_sam_utf, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        pass;

class osr_translate:
    CLS_gt_dtype=torch.long;
    # we now require explict convert unk first--- no suprise suprise
    @classmethod
    def translate_strict(cls,sam_utf,tdict):
        labelsraw = [];
        lens = [];
        for su in sam_utf:
            labelsraw.append([tdict[u] for u in su]);
            lens.append(len(su));
        return labelsraw,lens;
    @classmethod
    def translate(cls,sam_utf,tdict):
        labelsraw = [];
        lens = [];
        for su in sam_utf:
            labelsraw.append([tdict[u] if u in tdict else tdict[tUNK] for u in su]);
            lens.append(len(su));
        return labelsraw,lens;
    # well this maybe useful for various heads, so we put it here
    @classmethod
    def flatten_labels(cls,labelsraw,device):
        f=[];
        for l in labelsraw:
            f+=l;
        tenlabel = torch.tensor(f,device=device, dtype=cls.CLS_gt_dtype);
        return tenlabel;
