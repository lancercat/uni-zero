# anchor keys v6
from neko_2024_NGNW.common.ak6 import AK6
def get_v6_anchor(acfg,name,minT,maxT,h,w,minr,maxr,has_tfe,has_eglp,heads,override_col=None):
    acfg[name]={
        AK6.min_routing_ratio: minr,
        AK6.max_routing_ratio: maxr,
        AK6.target_size_wh: (w,h),
        AK6.padding_size: ((0, 0), (0, 0)),
        AK6.maxT: maxT,
        AK6.minT: minT,
        AK6.possible_rotation: [0],
        AK6.heads:heads,
        AK6.has_tfe:has_tfe,
        AK6.override_col:override_col
    }
    acfg[AK6.names].append(name);
    return acfg


def get_wna_v6_dcfg_1h1v1r_2_05_smol(bsf=24):
    return {
        AK6.profile_name:"wna_2_05_g5-1",
        AK6.names:["hori","rect","vert"],
        "hori":{
            AK6.training_range:(2,9999),
            AK6.batch_size:bsf,
            AK6.maxT: 32,
            AK6.minT: 1,
        },
        "rect":{
            AK6.training_range:(0.5,2),
            AK6.batch_size: bsf,
            AK6.maxT: 32,
            AK6.minT: 1,
        },
        "vert":{
            AK6.training_range:(-100,0.5),
            AK6.batch_size: bsf,
            AK6.maxT: 32,
            AK6.minT: 1,
        },
    };

def dan_only_core(acfg,name,minT,maxT,h,w,minr,maxr,type,override_collate=None):
    # use what as its rewards depends on head implementation
    heads={
        AK6.head_names:["main_"],
        "main_":{
            "type":type,
            "global_reward_weight":1,
            "local_reward_weight":1,
            "extra_loss_weight":0
        }
    };
    acfg=get_v6_anchor(acfg, name,minT,maxT,h,w,minr,maxr,True,True,heads,override_collate);
    return acfg;


# if its just dan, eglp=head specific length prediction
def dan_only(acfg,name,minT,maxT,h,w,minr,maxr,override_collate=None):
    return dan_only_core(acfg,name,minT,maxT,h,w,minr,maxr,"danish",override_collate);

def dan_only_nedmix(acfg,name,minT,maxT,h,w,minr,maxr,override_collate=None):
    # use what as its rewards depends on head implementation
    return dan_only_core(acfg,name,minT,maxT,h,w,minr,maxr,"dan-nedmix",override_collate);


def get_wna_v6_32_anchor_1h1v1r_nedmix():
    acfg={
        AK6.beacon_size_wh: (32, 32),
        AK6.names:[]};
    acfg=dan_only_nedmix(acfg,"hori_",1, 32, 24, 96, 1, 9999);
    acfg = dan_only_nedmix(acfg, "vert_", 1, 32, 96, 24, -9999, 1);
    acfg = dan_only_nedmix(acfg, "rect_", 1, 32 ,64,64 , 0.3, 3); # that's our best bet already:)
    return acfg;

def get_wna_v6_32_anchor_1h1v1r():
    acfg={
        AK6.beacon_size_wh: (32, 32),
        AK6.names:[]};
    acfg=dan_only(acfg,"hori_",1, 32, 24, 96, 1, 9999);
    acfg = dan_only(acfg, "vert_", 1, 32, 96, 24, -9999, 1);
    acfg = dan_only(acfg, "rect_", 1, 32 ,64,64 , 0.3, 3); # that's our best bet already:)
    return acfg;


def get_wna_v6_32_anchor_1h1v1r_nedmix_reg():
    acfg={
        AK6.beacon_size_wh: (32, 32),
        AK6.names:[]};
    acfg=dan_only_nedmix(acfg,"hori_",1, 32, 32, 128, 1, 9999);
    acfg = dan_only_nedmix(acfg, "vert_", 1, 32, 128, 32, -9999, 1);
    acfg = dan_only_nedmix(acfg, "rect_", 1, 32 ,64,64 , 0.3, 3); # that's our best bet already:)
    return acfg;



def get_wna_v6_dcfg_mono_32_128(bsf=24):
    return {
        AK6.profile_name:"wna_mono-eccv",
        AK6.names:["hori"],
        "hori":{
            AK6.training_range:(-9999,9999),
            AK6.batch_size:bsf,
            AK6.maxT: 32,
            AK6.minT: 1,
        }
    };

def get_wna_v6_32_128_mono_anchor_nedmix():
    acfg={
        AK6.beacon_size_wh: (32, 32),
        AK6.names:[]};
    acfg=dan_only_nedmix(acfg,"hori_",1, 32, 32, 128, -9999, 9999);
    return acfg;

