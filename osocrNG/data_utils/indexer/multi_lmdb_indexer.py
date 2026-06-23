import random
class neko_multi_lmdb_enumerator_rand_seed:
    def __init__(this,param):
        this.src_rng=random.Random(param["src_seed"]);
        this.idx_rng=random.Random(param["idx_seed"]);
        this.lengths=param["lengths"];
        # this.names=[i for i in this.length_dict];
        this.cnt=len(this.lengths);
        this.total=sum(this.lengths);
    def __next__(this):
        src=this.src_rng.randint(0,this.cnt-1);
        sam=this.idx_rng.randint(0,this.lengths[src]-1)
        return {"dsid":src,"descp":{"id":sam}}
