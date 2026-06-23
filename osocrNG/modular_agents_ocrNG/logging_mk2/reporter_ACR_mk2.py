import datetime
import time

from osocrNG.sptokens import tUNKREP


class reporter_core_ACR_mk2:
    def reset(this,name):
        this.name=name;
        this.tot=0.;

        this.correct_k = 0;
        this.tot_k = 0.;

        this.rej_hit=0.;
        this.rej_false=0.;
        this.gt_rej_tot=0.;

        this.corr_by_label={};

        this.sta_time=time.time();
    # some old benchmarks don't really include symbols in annotation, so in manyshot/gzsl mode predicted unks will be manually removed.
    # for deployment if you are confidence that your data does not include new character, do the same.
    # this strips unks in gt too IF strict is False, which is useful if you have bad annotations with all unintended characters.
    # this happens if you use mlt dataset and the annotators decide to put japanese and chinese into korean subsets.
    def __init__(this,name,remove_unk=True,strict=False):
        this.remove_unk=remove_unk;
        this.reset(name);
        this.strict=strict;

    def same(this, gt_tok,pr_tok,tdict):
        lgt = len(gt_tok);
        lpr = len(pr_tok);
        if (lgt != lpr):
            return 0;
        corr=1;
        for g,p in zip(gt_tok,pr_tok):
            if(tdict is None):
                if(g!=p):
                    corr=0;
                    break;
            else:
                if(tdict[g]!=tdict[p]):
                    corr=0;
                    break;# ca
        return corr;

    def has_unk(this,toks,tdict):
        for t in toks:
            if(t==tUNKREP):
                return True;
            if(t not in tdict):
                return True;
        return False;
    def hash_gt(this,gt_tok):
        return "".join(gt_tok);


    def record_one(this, gt_tok_,pr_tok_,tdict):
        if(this.remove_unk):
            # this is in case some testing test or training set has symbol annotation that is not covered by protocol---
            # making them unks that is not supposed to me measured --- in this caes we just remove them.
            # this is a problem with the MJST -> eng testing sets.
            pr_tok=[t for t  in pr_tok_ if (t != tUNKREP)];
            if(this.strict):
                gt_tok=gt_tok_;
                assert (this.has_unk(gt_tok,tdict)==0);
            else:
                gt_tok=[t for t  in gt_tok_ if (t != tUNKREP and t in tdict)];
        else:
            pr_tok=pr_tok_;
            gt_tok=gt_tok_;
            # print(gt_tok_);
            # if(this.tot==4):
            #     pass;
        if(this.has_unk(gt_tok,tdict)):
            this.gt_rej_tot+=1;
            # print(this.tot);
            # print(gt_tok);
            # print(len(tdict));
            # print(tdict);
            # fatal(9);
            if(this.has_unk(pr_tok,tdict)):
                this.rej_hit+=1;
            else:
                  pass;
        else:
            this.tot_k+=1;
            if(this.has_unk(pr_tok,tdict)):
                this.rej_false+=1;
            else:
                if(this.same(gt_tok,pr_tok,tdict)):
                    this.correct_k+=1;
                    lk=this.hash_gt(gt_tok);
                    # if(lk=="7_01"):
                    #     pass;
                    if( lk not in this.corr_by_label):
                        this.corr_by_label[lk]=0;
                    this.corr_by_label[lk]+=1;
                # else:
                #     if("02" in gt_tok[0] or "02" in pr_tok[0]):
                #         if(gt_tok[0].split("_")[0] == pr_tok[0].split("_")[0]):
                #             pass;

            pass;

        this.tot +=1;
    def record_batch(this,gt_tok_seqs, pr_tok_seqs, tdict):
        for g,p in zip(gt_tok_seqs,pr_tok_seqs):
            this.record_one(g,p,tdict);

    def report(this,eidx,bidx):
        if this.tot == 0:
            pass
        now=time.time();
        fps=this.tot/(now-this.sta_time);
        rd= {"Date": datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                "TEST": this.name,
                "Epoch": eidx,
                "Iter": bidx,
                "Total-K": this.tot_k,
                "ACR":this.correct_k /this.tot_k,
                "FPS":fps};
        if(this.gt_rej_tot):
            R=this.rej_hit/this.gt_rej_tot;
            P=this.rej_hit/max(1.,this.rej_hit+this.rej_false);
            F=2*R*P/max(0.0000001,R+P);
            rd["Recall"]=R;
            rd["Precision"]=P;
            rd["FScore"]=F;
            rd["Total-U"]=this.gt_rej_tot;
        return rd;
