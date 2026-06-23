import copy
import random

import numpy as np
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNKREP,tUNK

from neko_sdk.neko_framework_NG.libmeta.agents.utils import mk_id_dict
# in case you want to align subcenters.
class neko_gen_id_dict(ama):
    INPUT_tdict="tdict";
    OUTPUT_id_dict="id_dict";
    def  set_mod_io(this,iocvt_dict,modcvt_dict):
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.id_dict = this.register_output(this.OUTPUT_id_dict, iocvt_dict);
        pass;
    def  set_etc(this,params):
        pass;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        tdict = workspace.get(this.tdict);

        workspace.add(this.id_dict,mk_id_dict(tdict));
        return workspace,environment;
    @classmethod
    def get_agtcfg(cls,
        tdict,
        id_dict
    ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_tdict: tdict, cls.OUTPUT_id_dict: id_dict}, "modcvt_dict": {}}}




# sample all protos w.r.t the modality
class neko_add_modality(ama):
    INPUT_protos="protos";
    INPUT_proto_utf="proto_utf"; # Proto_utf needs to be unique.
    # if you want multi template, go with  "A_t1", "A_t2", etc,
    # this may change in 33x. Leave future to future.
    # and wire everything up in tdict (which is used to generate id_dict)
    # same goes with what ever else.
    # And do remember to map them in the tdict.
    INPUT_id_dict="id_dict"; # well we don't use tdict here bcs we want to be as aligned as possible.
    OUTPUT_sampled_protos="sampled_protos";

    def recover_ids(this,proto_utf):
        rd={};
        for i in range(len(proto_utf)):
            rd[proto_utf[i]]=i;
        return rd;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.id_dict = this.register_input(this.INPUT_id_dict, iocvt_dict);
        this.proto_utf = this.register_input(this.INPUT_proto_utf, iocvt_dict);
        this.protos = this.register_input(this.INPUT_protos, iocvt_dict);
        this.sampled_protos = this.register_output(this.OUTPUT_sampled_protos, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        id_dict = workspace.get(this.id_dict);
        proto_utf = workspace.get(this.proto_utf);
        protos = workspace.get(this.protos);
        u2l=this.recover_ids(proto_utf);
        sks=np.array(sorted(id_dict.keys()));
        aps=[];
        assert (sks.max()==len(sks)-1); # no skipped ids.
        for k in sks:
            for u in id_dict[k]:
                if(u in u2l):
                    aps.append(protos[u2l[u]]);
                else:
                    fatal(u+" should be in protos, but actually not....");
        workspace.add(this.sampled_protos,aps);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   id_dict, proto_utf, protos,
                   sampled_protos
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_id_dict: id_dict, cls.INPUT_proto_utf: proto_utf, cls.INPUT_protos: protos,
                           cls.OUTPUT_sampled_protos: sampled_protos}, "modcvt_dict": {}}}



# do not arm unknown unless you have all set up
# we do this so that different heads can arm  different sptokens before adding the unk
# note that the unk is embedding free, so there is no proto involved.
class make_sampled_plabel(ama):
    INPUT_id_dict="id_dict";
    INPUT_tdict="tdict";
    OUTPUT_plabel="plabel";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.id_dict = this.register_input(this.INPUT_id_dict, iocvt_dict);
        this.plabel = this.register_output(this.OUTPUT_plabel, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        id_dict = workspace.get(this.id_dict);
        sks=np.array(sorted(id_dict.keys()));
        pls=[];
        assert (sks.max()==len(sks)-1); # no skipped ids.
        for k in sks:
            pls.append(k);
        workspace.add(this.plabel,torch.tensor(pls));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   id_dict, tdict,
                   plabel
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_id_dict: id_dict, cls.INPUT_tdict: tdict, cls.OUTPUT_plabel: plabel},
            "modcvt_dict": {}}}