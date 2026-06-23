from osocrNG.names import default_ocr_variable_names as dvn
import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.ocr_modules.io.encdec import encode_core
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred
from osocrNG.names import default_ocr_variable_names as dvn
from neko_sdk.ocr_modules.sptokens import tUNK, tSPLIT
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
class neko_dan_len_making_agent(neko_module_wrapping_agent):
    INPUT_label_name = dvn.raw_label_name;
    INPUT_devind = "device_indicator";  # just give it whatever that indicates the device
    OUTPUT_tensor_gt_length_name = dvn.tensor_gt_length_name;
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.devind = this.register_input(this.INPUT_devind, iocvt_dict);
        this.label = this.register_input(this.INPUT_label_name, iocvt_dict);
        this.tensor_gt_length = this.register_output(this.OUTPUT_tensor_gt_length_name, iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        dev = workspace.get(this.devind).device;
        gt_length = [len(tokenize(s)) for s in  workspace.get(this.label)];
        tenlen = torch.tensor(gt_length, device=dev);
        workspace.add(this.tensor_gt_length, tenlen);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   devind, label_name,
                   tensor_gt_length_name
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_devind: devind, cls.INPUT_label_name: label_name,
                                                        cls.OUTPUT_tensor_gt_length_name: tensor_gt_length_name},
                                         "modcvt_dict": {}}}
# dan with decoupled length prediction
class neko_ccd_label_making_agent(neko_module_wrapping_agent):
    INPUT_label_name = dvn.raw_label_name;
    INPUT_tdict_name = dvn.tdict_name;
    INPUT_tensor_length=dvn.tensor_gt_length_name;
    OUTPUT_tensor_label_name=dvn.tensor_label_name;
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.tdict = this.register_input(this.INPUT_tdict_name, iocvt_dict);
        this.label=this.register_input(this.INPUT_label_name,iocvt_dict);
        this.tensor_length=this.register_input(this.INPUT_tensor_length,iocvt_dict);
        this.tensor_label = this.register_output(this.OUTPUT_tensor_label_name, iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        ls=workspace.get(this.tensor_length);
        label_batch=workspace.get(this.label);
        tdict=workspace.get(this.tdict);
        out = torch.zeros([ls.sum().item()], dtype=torch.long, device="cpu");
        sta=0;
        for i in range(0, len(label_batch)):
            ats=tokenize(label_batch[i]);
            l=len(ats);
            out[sta:l+sta] = torch.tensor([tdict[char] if char in tdict else tdict[tUNK]
                                        for char in ats]);
            sta=l+sta;
        workspace.add(this.tensor_label,out.to(ls.device));
        return workspace,environment;

    @classmethod
    def get_agtcfg(cls,
                   label_name, tdict_name, tensor_length,
                   tensor_label_name
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_label_name: label_name, cls.INPUT_tdict_name: tdict_name,
                           cls.INPUT_tensor_length: tensor_length, cls.OUTPUT_tensor_label_name: tensor_label_name},
            "modcvt_dict": {}}}


# legacy for object 310 and older;

class neko_label_making_agent(neko_module_wrapping_agent):
    OUTPUT_tensor_label_name=dvn.tensor_label_name;
    OUTPUT_tensor_gt_length_name=dvn.tensor_gt_length_name;
    INPUT_tdict_name = dvn.tdict_name;
    INPUT_label_name = dvn.raw_label_name;
    INPUT_devind="device_indicator"; # just give it whatever that indicates the device
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.devind=this.register_input(this.INPUT_devind,iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict_name, iocvt_dict);
        this.label=this.register_input(this.INPUT_label_name,iocvt_dict);
        this.tensor_gt_length=this.register_output(this.OUTPUT_tensor_gt_length_name,iocvt_dict);
        this.tensor_label = this.register_output(this.OUTPUT_tensor_label_name, iocvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        dev=workspace.get(this.devind).device;
        gt_label, gt_length = encode_core(
            workspace.get(this.tdict),
            workspace.get(this.label),
            device=dev);
        tenlen = torch.tensor(gt_length, device=dev);
        tenlab, _ = neko_flatten_NG_lenpred.inflate(gt_label, tenlen);
        workspace.add(this.tensor_label,tenlab);
        workspace.add(this.tensor_gt_length,tenlen);
        return workspace,environment;
def get_neko_label_making_agent(devind_name,label_name_name,tdict_name_name,tensor_gt_length_name_name,tensor_label_name_name):
    engine = neko_label_making_agent;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_devind: devind_name, engine.INPUT_label_name: label_name_name, engine.INPUT_tdict_name: tdict_name_name, engine.OUTPUT_tensor_gt_length_name: tensor_gt_length_name_name, engine.OUTPUT_tensor_label_name: tensor_label_name_name}, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_ccd_label_making_agent.print_default_setup_scripts()