# a meta is
class meta_define:
    PARAM_meta_root="meta_root";
    PARAM_sampler_capacity = "sampler_capacity"; # set to -1 if no sample needed.
    PARAM_sampler_sinfo_list= "sample_sinfo_list"; # what type of sinfo is enlisted? g
    def get_eps(this,data_prfx):
        pass;

    def __init__(self):
        pass;
# there is no point having a synth im db for meta--- you use the glyph or real im
# if you have to do that, bind that to an augmentation variant.
class sinfo_define:
    PARAM_db_root="data_root";
    PARAM_db_type="db_type"; # glyphpt, lmdb, embedding, etc




class task_define:
    PARAM_data_eps="data_eps"; # a list of endpoints, log probability required if routing is involved
    PARAM_meta_eps = "meta_eps"; # a list of endpoints, log probability required if routing is involved
