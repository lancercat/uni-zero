from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

import torch

from osocrNG.sptokens import tUNKREP,tUNK


class neko_inject_sp(ama):
    INPUT_in_tdict="in_tdict";
    INPUT_in_proto="in_proto";
    INPUT_in_utf_list="in_utf_list";
    OUTPUT_out_tdict="tdict_wsptoken";
    OUTPUT_out_proto="out_proto";
    OUTPUT_out_utf_list="out_utf_list";
    MOD_spmod="spmod";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.in_proto = this.register_input(this.INPUT_in_proto, iocvt_dict);
        this.in_tdict = this.register_input(this.INPUT_in_tdict, iocvt_dict);
        this.in_utf_list = this.register_input(this.INPUT_in_utf_list, iocvt_dict);
        this.out_proto = this.register_output(this.OUTPUT_out_proto, iocvt_dict);
        this.out_tdict = this.register_output(this.OUTPUT_out_tdict, iocvt_dict);
        this.out_utf_list = this.register_output(this.OUTPUT_out_utf_list, iocvt_dict);
        this.spmod = this.register_mod(this.MOD_spmod, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        in_proto = workspace.get(this.in_proto);
        in_tdict = workspace.get(this.in_tdict);
        in_utf_list = workspace.get(this.in_utf_list); # utf list is needed for 32x above, we need it to build plabel.

        proto, utf = environment.module_dict[this.spmod]();
        #proto, utf=[],[]; # force bypassing spinj
        offset = len(utf);
        ntdict = {};
        unkid = 0;
        for t in in_tdict:
            if (type(t) == str):
                ntdict[t] = in_tdict[t] + offset;
            else:
                ntdict[t + offset] = in_tdict[t];
                if (unkid <= t + offset):
                    unkid = t + offset + 1;
        for uid in range(offset):
            ntdict[uid] = utf[uid];
            ntdict[utf[uid]] = uid;
        assert (unkid not in ntdict);

        ntdict[unkid] = tUNKREP;
        ntdict[tUNK] = unkid;
        if(len(proto)):
            workspace.add(this.out_proto, torch.cat([proto, in_proto]));
        else:
            workspace.add(this.out_proto, in_proto);

        workspace.add(this.out_tdict, ntdict);
        workspace.add(this.out_utf_list,utf+in_utf_list);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   in_proto, in_tdict, in_utf_list,
                   out_proto, out_tdict, out_utf_list,
                   spmod
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_in_proto: in_proto, cls.INPUT_in_tdict: in_tdict,
                                                        cls.INPUT_in_utf_list: in_utf_list,
                                                        cls.OUTPUT_out_proto: out_proto,
                                                        cls.OUTPUT_out_tdict: out_tdict,
                                                        cls.OUTPUT_out_utf_list: out_utf_list},
                                         "modcvt_dict": {cls.MOD_spmod: spmod}}}



if __name__ == '__main__':
    neko_inject_sp.print_default_setup_scripts()