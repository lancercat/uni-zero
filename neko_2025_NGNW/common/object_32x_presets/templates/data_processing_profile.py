from neko_2025_NGNW.common.object_32x_presets.templates.dataspec import neko_basic_im_dataspec

class neko_human_filter_profile_meta:
    TYPE_AR_RNG="aspect_ratio_range";
    TYPE_T_RNG = "aspect_timestamp_range";
    TYPE_LANG_SET="language_set_filter";

    TYPE = "type"


class neko_ar_filter_profile(neko_human_filter_profile_meta):
    THIS_TYPE=neko_human_filter_profile_meta.TYPE_AR_RNG;
    AR_RNG="ar_range";
    @classmethod
    def get_ar_rng_filter_profile(cls,min=-999,max=9999):
        return {
            cls.TYPE:cls.THIS_TYPE,
            cls.AR_RNG: [min,max]
        }
# the specific on how an anchor (size template branch) is treated
class neko_img_process_profile_meta:
    TYPE_COLFE_col_fe="col_fe";
    TYPE_COLFE_null_fe = "null_fe";

    TYPE_AGGR_attn="attaggr";
    TYPE_AGGR_cam = "cam"; # length is predicted by head. this will just make N,T,P,C tensor.
    TYPE_AGGR_null = None; # no aggr,  the tasks/heads takes feature map towers as is.
                                       # this will be only used in contextual dependent decoders.
    TYPE_AGGR_squeeze_shorter = "squeeze_shorter_if_tie_assume_hori";

    TYPE_AGGR_roi = "single_roi"; # if it needs fake time will be handled by the builder.




class neko_img_process_profile(neko_img_process_profile_meta):
    TYPE_COLFE = "colfe";
    TYPE_AGGRS="aggrs";


    @classmethod
    def get_default_single_roi_process_dict(cls,col=True):
        cdict={};
        if(col):
            cdict[cls.TYPE_COLFE]=cls.TYPE_COLFE_col_fe;
        else:
            cdict[cls.TYPE_COLFE]=cls.TYPE_COLFE_null_fe;
        cdict[cls.TYPE_AGGRS]={
            cls.TYPE_AGGR_roi: cls.TYPE_AGGR_roi,
        }

        return cdict;
    # these are configured to have endpoints.
    @classmethod
    def get_default_multiroi_process_dict(cls,col=True,aggr_dict=None):
        if(aggr_dict is None):
            aggr_dict={
                cls.TYPE_AGGR_cam: cls.TYPE_AGGR_cam,
                # this is not repetition--- if you for some SimCLR/MoE reasons want to create two endpoints you can have another key
            }
        cdict={};
        if(col):
            cdict[cls.TYPE_COLFE]=cls.TYPE_COLFE_col_fe;
        else:
            cdict[cls.TYPE_COLFE]=cls.TYPE_COLFE_null_fe;
        cdict[cls.TYPE_AGGRS]=aggr_dict;
        return cdict;

# we collate samples bcs they sometimes they are of too diverse aspect ratios and need special collation
# and the collation could well be anything.

class neko_anchor_data_spec(neko_basic_im_dataspec):
    NMT="network_max_timestamp"; # gives the maximum timestamp hint to the module builder

    # no fork no nothing
    # if we want to have image classification, the size may change, hence the api goes with char, for now.
    # if we decide to merge, we merge.
    @classmethod
    def get_default_char_anchor_dict(cls,sz=32, name="char"):
        cdict=cls.get_default_template(sz,name);
        cdict[cls.NMT]=2; # aticipate 2 tokens, incase sptoken injection
        return cdict;

    @classmethod
    def get_default_word_anchor_dict(cls, szhw, name="word"):
        cdict = cls.get_default_template(szhw,name);
        cdict[cls.NMT] = 32;  # aticipate 32 tokens, incase sptoken injection
        return cdict;


class neko_anchor_process_profile_meta:
    TYPE_TEMPLATE_char="char";
    TYPE_TEMPLATE_word_routing_head = "word_routing_head"; # routes further within an anchor
    TYPE_TEMPLATE_word_non_routing="word_non_routing";
    TYPE_TEMPLATE_world_non_routing="world_non_routing"; # this in nerfed in 320 beta related releases

    TYPE_ITEM_ITEM="item";

    TYPE_SEQ_SQUEEZE="squeeze";
    TYPE_SEQ_DETR = "detr";


# meta and cmdr will mux this, with different configs
class neko_anchor_process_profile(neko_anchor_process_profile_meta):
    DATA_PRFX="data_prfx"; # asssume the data has be routed, so src does not concern it
    MOD_PRFX="mod_prfx";
    DATA_SPEC="data_spec";

    MOD_SPEC="mod_spec"; # will only be used after 35x when we figure out if we want mod control here or in the builder
    TYPE="type";
    TEMPLATE="template"; # for now, the builder will only look at template;
    FEAT_ENDPOINTS="feat_endpoints";
    ROI_ENDPOINTS="roi_endpoints";
    # COLFE_TYPE="col_fe_type";
    # AGGR_TYPE="aggr_type"; # None, "simp_att", ""
    @classmethod
    def get_default_char_anchor(cls,data_prfx,modprfx,size):
        return {
            cls.DATA_PRFX:data_prfx,
            cls.MOD_PRFX:modprfx,  # default process does not override
            # there is no fork here.
            cls.DATA_SPEC: neko_anchor_data_spec.get_default_char_anchor_dict(size,data_prfx),
            cls.MOD_SPEC:neko_img_process_profile.get_default_single_roi_process_dict(),
            cls.TEMPLATE:cls.TYPE_TEMPLATE_char,
            cls.FEAT_ENDPOINTS: [data_prfx],
            cls.ROI_ENDPOINTS: [data_prfx],
            cls.TYPE: cls.TYPE_ITEM_ITEM
        }

    @classmethod
    def get_default_word_anchor_no_routing(cls,data_prfx,modprfx,size):
        return {
            cls.DATA_PRFX:data_prfx,
            cls.MOD_PRFX:modprfx,  # default process does not override
            # there is no fork here.
            cls.DATA_SPEC:neko_anchor_data_spec.get_default_word_anchor_dict(size),
            cls.MOD_SPEC:neko_img_process_profile.get_default_single_roi_process_dict(),
            cls.TEMPLATE:cls.TYPE_TEMPLATE_word_non_routing,
            cls.FEAT_ENDPOINTS: [data_prfx],
            cls.ROI_ENDPOINTS: [data_prfx],
            cls.TYPE:cls.TYPE_SEQ_SQUEEZE
        }


class neko_datastream_processing_profile:
    ANCHOR_NAMES="anchor_names";
    ANCHOR_CFG_DICT="anchor_cfg_dicts";
    ROI_EPS="roi_eps"; # list all endpoints.
    FEAT_EPS="feat_eps";
    @classmethod
    def get_default_im_process(cls, data_prfx,modprfx, size):
        return {
            cls.ANCHOR_NAMES:[data_prfx],
            cls.ANCHOR_CFG_DICT:{
                data_prfx:neko_anchor_process_profile.get_default_char_anchor(data_prfx,modprfx,size) # this is not a bug, as no routing no forking.
            },
            cls.ROI_EPS: [data_prfx],
            cls.FEAT_EPS: [data_prfx]
        };
    @classmethod
    def get_default_word_process(cls, data_prfx,modprfx, size):
        return {
            cls.ANCHOR_NAMES:[data_prfx],
            cls.ANCHOR_CFG_DICT:{
                data_prfx:neko_anchor_process_profile.get_default_word_anchor_no_routing(data_prfx,modprfx,size) # this is not a bug, as no routing no forking.
            },
            cls.ROI_EPS: [data_prfx],
            cls.FEAT_EPS: [data_prfx]
        };
