# The NG sampler is no more a nn.Module
import os.path
import random

import numpy as np
import regex
import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.ocr_modules.sptokens import tUNK, tDC, tSPLIT
from neko_sdk.ocr_modules.sptokens import tUNKREP

from neko_sdk.log import warn
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

# it just samples
# we will eventually separate side info modality from sampler
# The sampler will sample based on gtdict.
# modality (be it radical seq, glyph image, descriptions, attribution stats, or whatever combinationts) will be indexed by tdict.

class neko_abstract_sideinfo_source(torch.nn.Module):
    PARAM_meta_path="meta_path";
    PARAM_case_sensitive="case_sensitive";
    def setup_meta(this, meta_args):
        this.EOS=0;
        this.case_sensitive = neko_get_arg(this.PARAM_case_sensitive,meta_args,False);
        this.meta_args=meta_args;
        this.masters_share = not this.case_sensitive;
        if(meta_args[this.PARAM_meta_path] is None):
            return ;
        try:
            meta=torch.load(os.path.join(meta_args[this.PARAM_meta_path],"vismeta.pt"));  # will be the future default.
        except:
            meta = torch.load(meta_args[this.PARAM_meta_path]);
            warn("triggered depcrated behaviour", meta_args[this.PARAM_meta_path]); # we are phasing out the old meta loading approach as we may slide in weird sideinformations.
        this.load_meta(meta);


    def debug(this,normpids,labels):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        protos=((torch.cat(normprotos, dim=-1).squeeze(0).squeeze(0)+1)*127.5).detach().cpu().numpy().astype(np.uint8);
        import cv2;
        cv2.imshow(labels,protos[:,:32*32])
        cv2.waitKey(0);

    def set_gbidict(this,oks):
        this.gbidict = {}
        gkset = {};
        gkcnt=0
        for k in oks:
            drk = this.label_dict[k];
            if (drk not in gkset):
                gkset[drk] = gkcnt;
                gkcnt += 1;
            this.label_dict[k] = gkset[drk];
            this.gbidict[k] = gkset[drk];
            if (drk == this.label_dict[k]):
                this.gbidict[gkset[drk]] = k;
            else:
                print("err");
                fatal("error");

    def __init__(this, meta_args):
        super().__init__();
        this.setup_meta(meta_args)

    def load_meta(this, meta):
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size, it's nightmare.
        list_character = list(meta["chars"]);
        this.aligned_characters = meta["achars"];
        # characters without shape is generally what you do now want to sample.
        this.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        this.character = list(meta["sp_tokens"]) + list_character;
        this.label_dict = meta["label_dict"];

        this.shaped_ids = set([this.label_dict[i] for i in this.shaped_characters]);

        this.sp_cnt = len(meta["sp_tokens"]);
        this.sp_tokens = meta["sp_tokens"];
        this.norm_protos=[];
        for p in meta["protos"][this.sp_cnt:]:
            if(p is not None and len(p.shape)==4):
                this.norm_protos.append(p[0].permute(1,2,0).contiguous());
            else:
                this.norm_protos.append(p);

        unk = this.label_dict[tUNK];
        # if the dict does not provide an specific unk token, set it to -1;
        for i, char in enumerate(this.character):
            # print(i, char)
            this.label_dict[char] = i;

        # shapeless unk shall be excluded
        if (unk < 0):
            this.label_set = set(this.label_dict.values()) - {unk};
        else:
            this.label_set = set(this.label_dict.values());

        this.prototype_cnt = -1;
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        this.masters = meta["master"];
        this.reduced_label_dict = {}
        this.reduced_bidict = {}

        kcnt = 0;
        kset = {};
        ks = []
        ls = []
        for k in this.label_dict:
            ks.append(k);
            ls.append(this.label_dict[k]);
        oks = [ks[i] for i in np.argsort(ls)];

        for k in oks:
            if (this.label_dict[k] in this.masters):
                drk = this.masters[this.label_dict[k]];
            else:
                drk = this.label_dict[k];
            if (drk not in kset):
                kset[drk] = kcnt;
                kcnt += 1;
            this.reduced_label_dict[k] = kset[drk];
            this.reduced_bidict[k] = kset[drk];
            if (drk == this.label_dict[k]):
                this.reduced_bidict[kset[drk]] = k;

        this.set_gbidict(oks);

        # Foes includes the characters looks like each other
        # but never share labels (They may actually have linguistic relationships...
        # Like yanderes in a broken relationship[x]).
        # This set helps implement ohem like minibatching on the huge labelset.
        # e.g. 'u' and 'ü'
        this.foes = meta["foes"];
        this.servants = meta["servants"];
        # union set of friend, harem and foe.
        this.related_proto_ids = meta["relationships"];

    def get_gplabel_and_dict_core(this, sappids, normpids, masters_share, use_sp=True, device="cpu"):
        if (use_sp):
            all_ids = sappids + normpids;
        else:
            all_ids = normpids;
        new_id = 0;
        plabels = [];
        labmap = {};
        bidict = {}
        gplabels = [];
        for i in all_ids:
            cha = this.aligned_characters[i];
            if (masters_share):
                vlab = this.masters[i];
            else:
                vlab = i;
            vcha = this.aligned_characters[vlab];
            if (vlab not in labmap):
                labmap[vlab] = new_id;
                # A new label
                new_id += 1;
                # sembs.append(this.semantic_embedding[vlab]);
            alab = labmap[vlab];
            plabels.append(alab);
            bidict[alab] = vcha;
            bidict[cha] = alab;
        plabels.append(new_id)
        bidict[tUNK] = new_id;
        bidict[tDC] = -1;

        if (this.masters_share):
            gbidict = this.reduced_bidict
        else:
            gbidict = this.gbidict

        for i in range(new_id):
            gplabels.append(gbidict[bidict[i]]);
        gplabels.append(gbidict[tUNK]);
        gbidict[gbidict[tUNK]] = "⑨";
        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        # set most special keys to "" if any.
        for s in sappids:
            bidict[s] = "";
        bidict[new_id] = tUNKREP;

        return torch.tensor(np.array(plabels), device=device), torch.tensor(np.array(gplabels), device=device), bidict, gbidict;

    def get_plabel_and_dictg(this,sappids,normpids,device="cpu"):
        return this.get_gplabel_and_dict_core(sappids,normpids,this.masters_share,device=device);
    def dump_all_impl(this, use_sp=True):
        if (use_sp):
            trsps = list(set([this.label_dict[i] for i in this.sp_tokens]));
        else:
            trsps = [];
        trchs = list(set([this.label_dict[i] for i in this.shaped_characters]));
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in trchs];
        plabels, gplabels, bidict, gbidict = this.get_plabel_and_dictg(trsps, trchs)
        return normprotos, plabels, gplabels, bidict, gbidict;

    def dump_all(this, use_sp=True):
        return this.dump_all_impl(use_sp);
    pass;


class neko_prototype_source_staticNG(neko_abstract_sideinfo_source):
    pass;


def dump_vis_prototype(meta_args,use_sp=False):
    return neko_prototype_source_staticNG(meta_args).dump_all(use_sp);


class neko_prototype_sampler_NG(neko_abstract_sideinfo_source):
    SPLIT=tSPLIT;
    # defines sampler
    def setup_sampler(this,sampler_args):
        this.max_batch_size = neko_get_arg("max_batch_size", sampler_args, 512);
        this.val_frac = neko_get_arg("val_frac", sampler_args, 0.8);
        this.neg_servant = neko_get_arg("neg_servant", sampler_args, True);
        this.rng = random.Random(neko_get_arg("seed", sampler_args, 9));

    def get_occured(this, text_batch):
        b = "";
        for _ in text_batch: b += _;
        return list(set(tokenize(b)));

    # No semb shit here, semb comes form meta, not sampler


    def grab_cluster(this,ch):
        chid=this.label_dict[ch];
        ret={chid};
        if this.masters_share:
            ret.add(this.masters[chid]);
            ret=ret.union(this.servants[this.masters[chid]]);
        return ret;

    def get_sampled_ids(this,plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * this.val_frac);
        cntval = min(this.max_batch_size - this.sp_cnt, cntval);
        trchs=set();
        related_chars_in_data=set();
        random.shuffle(plain_chars_in_data);
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval;
        for ch in plain_chars_in_data:
            if(ch not in this.label_dict):
                continue;
            new=this.grab_cluster(ch);
            ns=trchs.union(new);
            related_chars_in_data=related_chars_in_data.union(new);
            delta=len(ns)-len(trchs);
            if(delta<=remaining):
                trchs=ns;
                remaining-=delta;
        remaining=this.max_batch_size-this.sp_cnt-len(trchs);
        plain_charid_not_in_data=list(this.shaped_ids-related_chars_in_data);
        random.shuffle(plain_charid_not_in_data);
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if (remaining == 0):
                    break;
                if (this.neg_servant==False and this.masters[chid]!=chid):
                    continue;
                remaining-=1;
                trchs.add(chid);

        trsps=set([this.label_dict[i] for i in this.sp_tokens]);
        return trsps,trchs;

    def sample_charset_by_text(this,text_batch,use_sp=True,device="cpu"):
        plain_chars_in_data = this.get_occured(text_batch)
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        if(use_sp is not False):
            trsps=list(trsps);
        else:
            trsps=[];
        plabels,gplabels,tdicts,gtdicts=this.get_plabel_and_dictg(trsps,trchs,device=device)
        normprotos=[this.norm_protos[i-this.sp_cnt]for i in trchs];
        protoutfs=[this.gbidict[i] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,protoutfs,plabels,gplabels,tdicts,gtdicts;
    def __init__(this,param):
        super().__init__(param["meta_args"]);
        this.setup_sampler(param["sampler_args"]);
