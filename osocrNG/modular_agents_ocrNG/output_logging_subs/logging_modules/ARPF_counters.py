# Counter dedicated for NG routines
from torch import nn

from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_oswr_Attention_AR_counter_NG(nn.Module):
    def __init__(this, param):
        super().__init__();
        this.clear();
        this.display_string = neko_get_arg("display_string",param);
        this.case_sensitive = neko_get_arg("case_sensitive",param,False);
        this.debug=neko_get_arg("case_sensitive",)
    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.total_C = 0.
        this.total_W = 0.
        this.total_U=0.
        this.total_K=0.
        this.Ucorr=0.
        this.Kcorr=0.
        this.KtU=0.

    def add_iter(this,prdt_texts, labels,UNK="⑨"):
        if(labels is None):
            return ;
        start = 0
        start_o = 0
        if this.debug:
            for i in range(len(prdt_texts)):
                print(labels[i], "->-", prdt_texts[i]);

        this.total_samples += len(labels);
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|sadhkjashfkjasyhf') + prdt_texts[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('||sadhkjashfkjasyhf')]
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            cflag=int(labels[i] == prdt_texts[i]);
            this.correct = this.correct + cflag;
            if(labels[i].find(UNK)!=-1):
                this.total_U+=1.;
                this.Ucorr+=(prdt_texts[i].find(UNK)!=-1);
            else:
                this.total_K+=1.;
                this.Kcorr+=cflag;
                this.KtU+=(prdt_texts[i].find(UNK)!=-1);


    def show(this):
        print(this.display_string)
        if this.total_samples == 0:
            pass
        R=this.Ucorr / max(this.total_U,1);
        P=this.Ucorr / max(this.Ucorr + this.KtU,1);
        F=2*(R*P)/max(R+P,1.)
        return {"KACR":this.Kcorr / max(1,this.total_K),
                "R":R,"P":P,"F":F};


