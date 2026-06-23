import random
from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_multi_lmdb_enumerator_rand_seed_mk2:
    PARAM_seed="seed";
    PARAM_lengths="lengths";
    def __init__(this,param):
        this.rng=random.Random(neko_get_arg(this.PARAM_seed,param,9));
        this.lengths=param["lengths"];
        # this.names=[i for i in this.length_dict];
        this.cnt=len(this.lengths);
        this.total=sum(this.lengths);
    def __next__(this):
        src=this.rng.randint(0,this.cnt-1);
        sam=this.rng.randint(0,this.lengths[src]-1)
        return {"dsid":src,"descp":{"id":sam}}
