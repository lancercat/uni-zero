import copy
import random

import numpy as np
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNKREP,tUNK

from neko_sdk.neko_framework_NG.libmeta.agents.utils import mk_id_dict
# sample for each, or use a grounded sampler if alignment is needed
# will be covered by agentic mk2 staging at 33x or later
# the sampling will now sample on class level, which means it will eiter
## Take all centers of one class, or
## Take none of one class.
# No partial taking.


class neko_agentic_random_label_sampler(ama):
    INPUT_tdict_old = "tdict_old";
    OUTPUT_sampled_tdict="sampled_tdict";

    PARAM_seed="seed";
    PARAM_frac="frac";
    PARAM_max_capacity="max_capacity";

    def arm_new_dict(this,id_dict,tdict_old,sampled_ids):
        ntdict={};
        nid=0;
        for k in sampled_ids:
            ur=tdict_old[k];
            ntdict[nid] =ur;
            assert (ur in tdict_old);
            for uk in id_dict[k]:
                ntdict[uk]=nid;
            nid+=1;
        return ntdict;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tdict_old = this.register_input(this.INPUT_tdict_old, iocvt_dict);
        this.sampled_tdict = this.register_output(this.OUTPUT_sampled_tdict, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.frac = neko_get_arg(this.PARAM_frac, params);
        this.rng = random.Random(neko_get_arg(this.PARAM_seed, params));
        this.max_capacity=neko_get_arg(this.PARAM_max_capacity, params);
        pass;
    def sample_keys(this,srcdst:set,tot,sk,idd,frac):
        if(tot>=this.max_capacity):
            return srcdst,tot; # if we have enough, stop bothering.
        this.rng.shuffle(sk);
        scnt = min(int(len(sk) * frac), this.max_capacity-tot);
        for i in range(scnt):
            k=sk[i];
            if(k in srcdst):
                continue;
            tot += len(idd[k]);
            if (tot > this.max_capacity):
                break;
            srcdst.add(sk[i]);
        return srcdst,tot;


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict_old = workspace.get(this.tdict_old);
        idd=mk_id_dict(tdict_old);
        sk=list(idd.keys());
        ssk=set();
        ssk,tot=this.sample_keys(ssk,0,sk,idd,this.frac);
        ssk=list(sorted(ssk));
        sampled_tdict=this.arm_new_dict(idd,tdict_old,ssk);
        workspace.add(this.sampled_tdict,sampled_tdict);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tdict_old,
                    sampled_tdict,
                   frac,seed=9,max_capacity=512
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_tdict_old: tdict_old,
                           cls.OUTPUT_sampled_tdict: sampled_tdict}, cls.PARAM_frac: frac,cls.PARAM_seed:seed,cls.PARAM_max_capacity:max_capacity, "modcvt_dict": {}}}


class neko_agentic_label_grounded_sampler(neko_agentic_random_label_sampler):
    INPUT_label_tokenized="raw_labels_tokenized";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.label_tokenized = this.register_input(this.INPUT_label_tokenized, iocvt_dict);
        pass;
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        label_tokenized = workspace.get(this.label_tokenized);
        tdict_old = workspace.get(this.tdict_old);
        idd=mk_id_dict(tdict_old);
        lk = set([tdict_old[i] for i in sum(label_tokenized,[]) ]); # ids from label
        rsk = set(idd.keys()).difference(lk); # ids not in label

        ssk = set();
        ssk, tot = this.sample_keys(ssk, 0, list(lk), idd, this.frac); # get this.frac of characters in the label.
        ssk, tot = this.sample_keys(ssk, tot, list(rsk), idd, 1);  # get this.frac of characters not in the label to fill the batch.

        ssk=list(sorted(ssk));
        sampled_tdict=this.arm_new_dict(idd,tdict_old,ssk);
        workspace.add(this.sampled_tdict,sampled_tdict);
        return workspace, environment;
    @classmethod
    def get_agtcfg(cls,
                   label_tokenized, tdict_old,
                   sampled_tdict,
                   frac, max_capacity, seed=9
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_label_tokenized: label_tokenized, cls.INPUT_tdict_old: tdict_old,
                           cls.OUTPUT_sampled_tdict: sampled_tdict}, cls.PARAM_frac: frac,
            cls.PARAM_max_capacity: max_capacity, cls.PARAM_seed: seed, "modcvt_dict": {}}}

class neko_agentic_label_grounded_sampler_multi(neko_agentic_random_label_sampler):
    INPUT_list_label_tokenized="raw_labels_tokenized";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.list_label_tokenized = this.register_input(this.INPUT_list_label_tokenized, iocvt_dict);
        pass;
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        label_tokenized = [];
        for lsrc in this.list_label_tokenized:
            label_tokenized+=workspace.get(lsrc);

        tdict_old = workspace.get(this.tdict_old);
        idd=mk_id_dict(tdict_old);
        lk = set(); # ids from label
        for i in sum(label_tokenized, []):
            if(i in tdict_old): # if there are oov already in training set. (mostly mjst)
                lk.add(tdict_old[i] );
        rsk = set(idd.keys()).difference(lk); # ids not in label

        ssk = set();
        ssk, tot = this.sample_keys(ssk, 0, list(lk), idd, this.frac); # get this.frac of characters in the label.
        ssk, tot = this.sample_keys(ssk, tot, list(rsk), idd, 1);  # get this.frac of characters not in the label to fill the batch.

        ssk=list(sorted(ssk));
        sampled_tdict=this.arm_new_dict(idd,tdict_old,ssk);
        workspace.add(this.sampled_tdict,sampled_tdict);
        return workspace, environment;
    @classmethod
    def get_agtcfg(cls,
                   list_label_tokenized, tdict_old,
                   sampled_tdict,
                   frac, max_capacity, seed=9
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_list_label_tokenized: list_label_tokenized, cls.INPUT_tdict_old: tdict_old,
                           cls.OUTPUT_sampled_tdict: sampled_tdict}, cls.PARAM_frac: frac,
            cls.PARAM_max_capacity: max_capacity, cls.PARAM_seed: seed, "modcvt_dict": {}}}

if __name__ == '__main__':
    # only cas

    neko_agentic_label_grounded_sampler.print_default_setup_scripts()