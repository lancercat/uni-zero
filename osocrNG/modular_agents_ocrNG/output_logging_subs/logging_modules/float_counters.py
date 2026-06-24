class float_counter_mean:
    def __init__(this, display_interval):
        this.display_interval = display_interval;
        this.total_iters = 0.;
        this.termsum={};
    def mkterm(this,dic,prfx=""):
        for k in dic:
            if(type(dic[k])==dict):
                this.mkterm(dic[k],prfx+k);
            else:
                if prfx+k not in this.termsum:
                    this.termsum[prfx+k] = 0;
                this.termsum[prfx+k] += float(dic[k]);

    def add_iter(this, terms):
        this.total_iters += 1;
        this.mkterm(terms);

    def clear(this):
        this.total_iters = 0;
        this.termsum={};
    def show(this,clearstat=True):
        if(clearstat):
            this.clear();
        retterms={};
        for k in this.termsum:
            term = this.termsum[k] / this.total_iters if this.total_iters > 0 else 0;
            retterms[k]=term;
        this.clear();
        return retterms;

