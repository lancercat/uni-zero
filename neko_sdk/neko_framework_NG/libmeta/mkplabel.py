import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNK,tUNKREP
#usually, this goes
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_plabel_mker(ama):
    INPUT_tdict="tdict";
    INPUT_utfs="utfs";
    INPUT_device_indicator="dev_ind";
    OUTPUT_plabel_collated="collated_plabel";
    PARAM_cntrless="centerless";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.device_indicator = this.register_input(this.INPUT_device_indicator, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.utfs = this.register_input(this.INPUT_utfs, iocvt_dict);
        this.plabel_collated = this.register_output(this.OUTPUT_plabel_collated, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.cntrless = neko_get_arg(this.PARAM_cntrless, params,[tUNK]);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        utfs = workspace.get(this.utfs);
        if(this.device_indicator is not None):
            device_indicator = workspace.get(this.device_indicator);
            dev=device_indicator.device;
        else:
            dev=workspace.device;
        plabel=torch.tensor([int(tdict[k])  for k in utfs+this.cntrless]).to(device=dev)
        workspace.add(
            this.plabel_collated,plabel
        );  # well this thing does not check if utf matches protos. Up to you...
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   device_indicator, tdict, utfs,
                   plabel_collated,
                   cntrless
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_device_indicator: device_indicator, cls.INPUT_tdict: tdict, cls.INPUT_utfs: utfs,
                           cls.OUTPUT_plabel_collated: plabel_collated}, cls.PARAM_cntrless: cntrless,
            "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_plabel_mker.print_default_setup_scripts()