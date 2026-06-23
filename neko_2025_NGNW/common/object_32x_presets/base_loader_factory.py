import tqdm
import os
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.ocr_multi_lmdb_size_balanced_fetch import im_text_multisource_lmdb_scale_balanced_fetch
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.ordered_lmdb_factory import single_ordered_lmdb_factory

from osocrNG.data_utils.common_data_presets_mk5.sideinfo.holders.pt_single_info_holder import neko_sideinfo_longid_factory
from osocrNG.data_utils.common_data_presets_mk5.sideinfo.holders.pt_holder import neko_sideinfo_diskloader_pt_factory
from osocrNG.data_utils.common_data_presets_mk5.sideinfo.holders.database_holder import neko_sideinfo_diskloader_lmdb_factory

from neko_2025_NGNW.common.agents.training_and_testing.neko_320_training_loader_agent import neko_320_training_loader_agent
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN,P
from osocrNG.data_utils.common_data_presets_mk5.data.collators.list_im_collators import neko_im_text_collate_factory,neko_semins_collate_factory

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awa_nograd
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent,neko_symbol_link_alot_agent_weak
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

from neko_2025_NGNW.common.object_32x_presets.templates.data_stream_profiles import neko_datastream_profile, \
    neko_augcol_template, neko_db_profile, neko_diskcol_template,neko_testing_load_balance_profile;
# from neko_2025_NGNW.common.object_32x_presets.templates.lsct_stream_profiles import neko_lsct_repeater_template,neko_lsct_char_random_template

from neko_2025_NGNW.common.object_32x_presets.templates.meta_stream_profiles import neko_meta_stream_profile,neko_meta_sampler,neko_meta_repo,neko_sideinfo_repo

# activelearning gonna wait after we have blackwell together with better io. Prefetching is still the life line here. So we cannot afford gradient
from osocrNG.data_utils.common_data_presets_mk5.data.presets.abiaugfix import abi_fixed_training
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

from neko_2025_NGNW.common.object_32x_presets.meta_templates.decoupled_sampling import training_sampled_meta_agt_factory,testing_meta_agt_factory
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory
# from neko_2025_NGNW.common.object_32x_presets.lsct.repeater import neko_lsct_shadow_factory
# from neko_2025_NGNW.common.object_32x_presets.lsct.rand_char_lmdb import neko_lsct_rand_char_glyphdb_factory
# for testing, meta need to be loaded first for prototype caching---
# for training, data needs to be loaded first so we can sample meta---
# that's why this thing is so twisted.


# that's a huge weight off shoulder bcs we don't need to worry wrapping queues in a service, which is a messy hell :)
# the collate here does not really "collate" it --- it just collate thumbnails
# this is a huge class--- we will figure out how to decompose later.
class neko_32x_data_loader_agent_factory_core(virtual_agt_factory):
    COLF = neko_im_text_collate_factory;
    MCOLF=neko_semins_collate_factory;

    SINFO_GLYPHPTF=neko_sideinfo_diskloader_pt_factory;
    SINFO_IMDBF=neko_sideinfo_diskloader_lmdb_factory;
    SINFO_LONG_ID=neko_sideinfo_longid_factory;

    def setup(this):
        this.sinfo_repo_pt= this.SINFO_GLYPHPTF();
        this.sinfo_repo_imdb= this.SINFO_IMDBF();
        this.sinfo_repo_longid= this.SINFO_LONG_ID();


    def arm_one_sinfo_loader(this,acfg, cur_sinfo_cfg,meta_prfx,random, seed):
        type = cur_sinfo_cfg[neko_sideinfo_repo.TYPE];
        sinfo_name = cur_sinfo_cfg[neko_sideinfo_repo.NAME];
        if (type == neko_sideinfo_repo.TYPE_GLYPH_PT):
            this.sinfo_repo_pt.arm_agts(acfg, {
                this.sinfo_repo_pt.META_PRFX: meta_prfx,
                this.sinfo_repo_pt.SIDE_INFO_PRFX: sinfo_name,
            }, {}, {this.sinfo_repo_pt.AGT_PRFX: sinfo_name}, {
                this.sinfo_repo_pt.PARAM_sideinfodb_root: cur_sinfo_cfg[neko_sideinfo_repo.DBCFG],
                                        });
            this.arm_im_meta_collate(
                acfg, sinfo_name, P(sinfo_name, "collate"), cur_sinfo_cfg[neko_sideinfo_repo.TEMPLATE], cv2=True
            );

        elif (type == neko_sideinfo_repo.TYPE_IM_DB):
            this.sinfo_repo_imdb.arm_agts(acfg, {
                this.sinfo_repo_imdb.META_PRFX: meta_prfx,
                this.sinfo_repo_imdb.SIDE_INFO_PRFX: sinfo_name,
            }, {}, {this.sinfo_repo_imdb.AGT_PRFX: sinfo_name}, {
                this.sinfo_repo_imdb.PARAM_sideinfodb_root: cur_sinfo_cfg[neko_sideinfo_repo.DBCFG],
                this.sinfo_repo_imdb.PARAM_k: cur_sinfo_cfg[neko_sideinfo_repo.K],
                this.sinfo_repo_imdb.PARAM_random: random,
                this.sinfo_repo_imdb.PARAM_force_grey: cur_sinfo_cfg[neko_sideinfo_repo.FORCE_GREY],
                this.sinfo_repo_imdb.PARAM_seed:seed});
            this.arm_im_meta_collate(
                acfg, sinfo_name, P(sinfo_name, "collate"), cur_sinfo_cfg[neko_sideinfo_repo.TEMPLATE], cv2=True
            );
        # no side info, which means the downstream task will just use tdict and utf to fetch from
        # if you need the network to handle sideinfo mgmt (pure data driven)
        elif (type == neko_sideinfo_repo.TYPE_LONG_ID):
            this.sinfo_repo_longid.arm_agts(acfg, {
                this.sinfo_repo_longid.META_PRFX: meta_prfx,
                this.sinfo_repo_longid.SIDE_INFO_PRFX: sinfo_name,
            }, {}, {this.sinfo_repo_longid.AGT_PRFX: sinfo_name}, {
                this.sinfo_repo_longid.PARAM_sideinfodb_root: cur_sinfo_cfg[
                    neko_sideinfo_repo.DBCFG]
            });
        else:
            fatal("wut");
        return acfg;
    def arm_im_meta_collate(this,acfg,prfx,aprfx,data_template,cv2=False):
        acfg=this.COLF.arm_collate_mvn(acfg, {
            this.COLF.DATA_PRFX: prfx
        }, {}, {this.COLF.AGT_PRFX: aprfx},
                              {
                                this.COLF.PARAM_padding_size_hw:[0,0],
                               this.COLF.PARAM_img_size_max_area:128*128,
                               this.COLF.PARAM_is_np:cv2
                               });
        return acfg;

    # aug cols
    def arm_collate(this,acfg,prfx,aprfx,data_template,cv2=False):

        acfg=this.COLF.arm_collates(acfg, {
            this.COLF.DATA_PRFX: prfx
        }, {}, {this.COLF.AGT_PRFX: aprfx},
                              {this.COLF.PARAM_thumb_size_hw: data_template[neko_diskcol_template.THUMB_SIZE],
                               this.COLF.PARAM_padding_size_hw: data_template[neko_diskcol_template.THUMB_PAD],
                               this.COLF.PARAM_img_size_max_area:data_template[neko_diskcol_template.RAW_IM_MAX_AREA],
                               this.COLF.PARAM_is_np:cv2
                               });
        return acfg;
    def arm_msk_collate(this,acfg,prfx,aprfx,data_template,cv2=False):
        acfg=this.MCOLF.arm_collate_mask_optional(acfg, {
            this.MCOLF.DATA_PRFX: prfx
        }, {}, {this.MCOLF.AGT_PRFX: aprfx},
                              {this.MCOLF.PARAM_thumb_size_hw: data_template[neko_diskcol_template.THUMB_SIZE],
                               this.MCOLF.PARAM_padding_size_hw: data_template[neko_diskcol_template.THUMB_PAD],
                               this.MCOLF.PARAM_img_size_max_area:data_template[neko_diskcol_template.RAW_IM_MAX_AREA],
                               this.MCOLF.PARAM_is_np:cv2
                               });
        return acfg;


    def arm_noaugcol_agt(this,acfg,raw_prfx,col_prfx,data_template,seed):
        acfg=awa.append_agent_to_cfg(
            acfg,P(col_prfx,"cp_imgt"),neko_symbol_link_alot_agent.get_agtcfg(
                [VN.IM_raw(raw_prfx),VN.UTF(raw_prfx),VN.GT_TOK_UTF(raw_prfx),VN.SAM_ORIG(raw_prfx)],
                [VN.IM_raw(col_prfx),VN.UTF(col_prfx),VN.GT_TOK_UTF(col_prfx),VN.SAM_ORIG(col_prfx)]));
        acfg=awa.append_agent_to_cfg(
            acfg,P(col_prfx,"cp_msk"),neko_symbol_link_alot_agent_weak.get_agtcfg(
                [VN.GT_INSSEG_MSK(raw_prfx),VN.GT_INSSEG_UTF(raw_prfx),VN.GT_SEMSEG_CBM_MSK(raw_prfx),VN.GT_SEMSEG_CBM_UTF(raw_prfx)],
                [VN.GT_INSSEG_MSK(col_prfx), VN.GT_INSSEG_UTF(col_prfx),  VN.GT_SEMSEG_CBM_MSK(col_prfx), VN.GT_SEMSEG_CBM_UTF(col_prfx)]));

        acfg = this.arm_collate(acfg,col_prfx,col_prfx,data_template);
        acfg = this.arm_msk_collate(acfg,col_prfx,col_prfx,data_template)
        return acfg;

    def __init__(this, dataroot):
        this.data_root = dataroot;
        this.setup()

        pass;

    pass;

class neko_32x_testing_meta_loader_agent_factory(neko_32x_data_loader_agent_factory_core):
    MLDRF=testing_meta_agt_factory;
    DLDRF=single_ordered_lmdb_factory;

    def setup(this):
        super().setup();
        this.mldrf=this.MLDRF();
        this.dldrf=this.DLDRF();

    def arm_one_diskldr_agents(this,acfg,datastream_profile_dict,data_root,seed):

        for k in datastream_profile_dict:
            cur_stream_profile_dict=datastream_profile_dict[k];
            dbcfg=cur_stream_profile_dict[neko_datastream_profile.DB];
            dprfx=dbcfg[neko_db_profile.NAME];
            srcs=dbcfg[neko_db_profile.SRCS];
            force_grey=dbcfg[neko_db_profile.FORCE_GRAY];
            snames=list(srcs.keys());
            assert (len(snames)==1); # one a time. this is testing.

            acfg=this.dldrf.arm_data_repo(
                acfg,{this.dldrf.DATA_PRFX:dprfx},{},
                {this.dldrf.AGT_PRFX:P("diskldr",dprfx)},
                {
                    this.dldrf.PARAM_batch_size:cur_stream_profile_dict[neko_datastream_profile.BALCFG][neko_testing_load_balance_profile.BATCH_SIZE],
                    this.dldrf.PARAM_split_pattern:dbcfg[neko_db_profile.TOKENIZER_KEY], # i know this looks ugly, but bear with this.
                    this.dldrf.PARAM_root:srcs[snames[0]],
                    this.dldrf.PARAM_force_grey:force_grey
                 });
            augcols=cur_stream_profile_dict[neko_datastream_profile.AUGCOLS];
            for aname in augcols:
                augcol_cfg=augcols[aname]
                aug_prfx=P(dprfx,aname);
                acfg = this.arm_noaugcol_agt(acfg, dprfx, aug_prfx, augcol_cfg[neko_augcol_template.DSKCOL_TEMPLATE], seed);

        return acfg;

    def arm_one_testing_meta_loader(this,acfg,metastream_profile_dict,data_root,seed):
        for k in metastream_profile_dict:
            cur_stream_profile_dict=metastream_profile_dict[k];
            mre_cfg=cur_stream_profile_dict[neko_meta_stream_profile.META_REPO];
            acfg=this.mldrf.arm_agt_core({},{
                this.mldrf.DATA_PRFX_local:cur_stream_profile_dict[neko_meta_stream_profile.META_ENDPOINTS][0]
            },acfg,k,{
                this.mldrf.PARAM_meta_root:mre_cfg[neko_meta_repo.PATH]
            });
        return acfg;

    # testing has nothing to do with lsct...
    # for testing, its metaloading -> batched prototype caching -> testing with testing data.
    # and thus we need to return 3 agent groups...
    def get_testing_bundle(this,dprofile_dict,mprofile_dict,pcfg:neko_platform_cfg):
        # everything single threaded, no armdict needed
        diskldrs=awa.empty();
        metaldrs=awa.empty();
        sinfo_ldrs=awa.empty();

        this.arm_one_diskldr_agents(diskldrs,dprofile_dict,pcfg.data_root,None);
        this.arm_one_testing_meta_loader(metaldrs,mprofile_dict,pcfg.data_root,None);
        for k in mprofile_dict:
            for s in mprofile_dict[k][neko_meta_stream_profile.SIDE_INFO_REPOS]:
                sr=mprofile_dict[k][neko_meta_stream_profile.SIDE_INFO_REPOS][s];
                sinfo_ldrs=this.arm_one_sinfo_loader(sinfo_ldrs,sr,mprofile_dict[k][neko_meta_stream_profile.META_ENDPOINTS][0],False,None); # testing does not use random proto.
                pass;
        return diskldrs,metaldrs,sinfo_ldrs;



class neko_32x_training_data_loader_agent_factory(neko_32x_data_loader_agent_factory_core):

    # params
    # endpoint management is still user responsibility till now.
    PARAM_datastream_profile_dict="datastream_profile_dict"; # include data, augmentation, collation
    PARAM_metastream_profile_dict="metastream_profile_dict"; # include specifiction of meta/sampling/sideinfoloading
    PARAM_task_profile_dict="task_profile_dict"; # include various things you do for extracting/aggrating data and compare to meta.

    # consts on how many parallels are allowed
    NUM_diskldr = 1;
    NUM_augcol = 9999;
    NUM_meta = 1;
    NUM_sinfo = 1;

    NUM_lsct = 3; # lsct is aggressive on ram...

    SEEDS=[9, # strongest ice fairy
           39, # short hand uss quincy
           42, # the answer
           0xca39, 0xca71, # uss quincy
           114514, 666, 110, # bad numbers
           1000000009, 1000000007, # primes
            0xbb64, 0xca73, # other ships
           1337,12345,0xDEADBEEF,314159265,271828182, # for excessive high core count.

           # --- Expanded Mathematical Constants ---
           161803398,  # Golden Ratio (phi) * 10^8
           141421356,  # Square root of 2 * 10^8
           57721566,  # Euler-Mascheroni constant * 10^8

           # --- Cryptographic & System Seeds ---
           0x5F3759DF,  # Fast inverse square root constant (Quake III)
           0xFEEDFACE,  # Mach-O binary magic number
           0xCAFEBABE,  # Java class file magic number
           2147483647,  # Mersenne prime (2^31 - 1)

           # --- Pop Culture & Lore ---
           451,  # Fahrenheit 451 / Ray Bradbury
           101,  # Vault 101 / Room 101
           117,  # Spartan-117 (Master Chief)
           1701,  # NCC-1701 (Enterprise)
           2049,  # Blade Runner 2049

           # --- Robust Primes (Safe Primes) ---
           1073741823,  # 2^30 - 1
           3221225473  # Large prime < 2^32
           ][:8][:os.cpu_count()]; # to not oom alvis...


    LDR_AGT=neko_320_training_loader_agent;

    AUG_ABIF=abi_fixed_training;


    IM_TEXT_DLDRF=im_text_multisource_lmdb_scale_balanced_fetch;
    # PANOPTIC_DLDRF = panoptic_multisource_lmdb_scale_balanced_fetch;

    MLDRF=training_sampled_meta_agt_factory;
    # LSCT_SHAD=neko_lsct_shadow_factory;
    # LSCTC_RAND=neko_lsct_rand_char_glyphdb_factory;

    DATA_EPS="data_eps";
    META_EPS = "meta_eps";




    def arm_abiaugcol_agt(this,acfg,raw_prfx,col_prfx,data_template,seed):
        acfg=this.AUG_ABIF.arm_agts(acfg,
                                {this.AUG_ABIF.DATA_PRFX:raw_prfx,this.AUG_ABIF.DATA_PRFX_AUGED_IMG:col_prfx},
                                {},
                                {this.AUG_ABIF.AGT_PRFX:col_prfx}, {  this.AUG_ABIF.PARAM_SEED:seed});
        acfg = this.arm_collate(acfg,col_prfx,col_prfx,data_template);
        return acfg;



    def arm_one_diskldr_agents(this,acfg,datastream_profile_dict,data_root,seed):

        for k in datastream_profile_dict:
            cur_stream_profile_dict=datastream_profile_dict[k];
            dbcfg=cur_stream_profile_dict[neko_datastream_profile.DB];

            if(dbcfg[neko_db_profile.TYPE]=="im_text_lmdb"):
                acfg=this.im_text_dldrf.arm_data_repo(
                    acfg,{this.im_text_dldrf.DATA_PRFX:k},{},
                    {this.im_text_dldrf.AGT_PRFX:P("diskldr",k)},
                    {
                        this.im_text_dldrf.PARAM_SEED:seed,
                        this.im_text_dldrf.PARAM_split_pattern:dbcfg[neko_db_profile.TOKENIZER_KEY], # i know this looks ugly, but bear with this.
                        this.im_text_dldrf.PARAM_dataloader_profile: cur_stream_profile_dict
                     });
            elif(dbcfg[neko_db_profile.TYPE]=="panoptic_lmdb"):
                acfg=this.panoptic_dldrf.arm_data_repo(
                    acfg, {this.panoptic_dldrf.DATA_PRFX: k}, {},
                    {this.panoptic_dldrf.AGT_PRFX: P("diskldr", k)},
                    {
                        this.panoptic_dldrf.PARAM_SEED: seed,
                        this.panoptic_dldrf.PARAM_split_pattern: dbcfg[neko_db_profile.TOKENIZER_KEY],
                        # i know this looks ugly, but bear with this.
                        this.panoptic_dldrf.PARAM_dataloader_profile: cur_stream_profile_dict
                    });
            else:
                fatal("wut");
        return acfg;

    def arm_one_augcol_agt(this,acfg,datastream_profile_dict,data_root,seed):
        for k in datastream_profile_dict:
            cur_stream_profile_dict=datastream_profile_dict[k];
            augcols=cur_stream_profile_dict[neko_datastream_profile.AUGCOLS];
            for aname in augcols:
                augcol_cfg=augcols[aname]
                aug_prfx=P(k,aname);
                if(augcol_cfg[neko_augcol_template.AUG_TYPE]==neko_augcol_template.TYPE_NOAUG):
                    acfg=this.arm_noaugcol_agt(acfg,k,aug_prfx,augcol_cfg[neko_augcol_template.DSKCOL_TEMPLATE],seed);
                else:
                    acfg=this.arm_abiaugcol_agt(acfg,k,aug_prfx,augcol_cfg[neko_augcol_template.DSKCOL_TEMPLATE],seed);
        return acfg;

    def arm_one_training_meta_loader(this,acfg,metastream_profile_dict,data_root,seed):
        for k in metastream_profile_dict:
            cur_stream_profile_dict=metastream_profile_dict[k];
            sam_cfg=cur_stream_profile_dict[neko_meta_stream_profile.META_SAMPLER];
            mre_cfg=cur_stream_profile_dict[neko_meta_stream_profile.META_REPO];

            acfg=this.mldrf.arm_agt_core({},{
                this.mldrf.DATA_PRFX_unsampled:k,
                this.mldrf.DATA_PRFX_refdata: sam_cfg[neko_meta_sampler.REF_DATA_prfx_list],
                this.mldrf.DATA_PRFX_local: neko_meta_sampler.get_dft_sampled_name(k)
            },acfg,P(k,"sampler"),{
                this.mldrf.PARAM_SEED:seed,
                this.mldrf.PARAM_frac: sam_cfg[neko_meta_sampler.FRAC],
                this.mldrf.PARAM_max_proto_capacity: sam_cfg[neko_meta_sampler.CAPACITY],
                this.mldrf.PARAM_meta_root:mre_cfg[neko_meta_repo.PATH]
            });
        return acfg;


    def arm_one_training_sinfo_loader(this,acfg,metastream_profile_dict,data_root,seed):
        for k in metastream_profile_dict:
            cur_stream_profile_dict=metastream_profile_dict[k];
            sre_cfgs = cur_stream_profile_dict[neko_meta_stream_profile.SIDE_INFO_REPOS];
            sampled_meta_prfx= neko_meta_sampler.get_dft_sampled_name(k);
            for s in sre_cfgs:
                cur_sinfo_cfg=sre_cfgs[s];
                this.arm_one_sinfo_loader(acfg,cur_sinfo_cfg,sampled_meta_prfx,True,seed);
        return acfg;

    # other type of lsct modules will be added later.
    def arm_one_shadow_lsct(this,acfg,cur_lsct_profile_dict,name,data_root,seed):
        type=cur_lsct_profile_dict[neko_lsct_repeater_template.TYPE]
        this.lsctf.arm_agt_core({},{
            this.lsctf.DATA_PRFX_local:name,
            this.lsctf.DATA_PRFX_ref:cur_lsct_profile_dict[neko_lsct_repeater_template.REF_DATA_PRFX]
        },acfg,type,{
            this.lsctf.PARAM_size:cur_lsct_profile_dict[neko_lsct_repeater_template.FNT_SIZE],
            this.lsctf.PARAM_im_root:cur_lsct_profile_dict[neko_lsct_repeater_template.BG_DB],
            this.lsctf.PARAM_fnt_root:cur_lsct_profile_dict[neko_lsct_repeater_template.FNT_ROOT],
            this.lsctf.PARAM_fnt_idx_file: cur_lsct_profile_dict[neko_lsct_repeater_template.FNT_IDX_PATH],
            this.lsctf.PARAM_seed:seed,
            this.lsctf.PARAM_charspace: cur_lsct_profile_dict[neko_lsct_repeater_template.SPACE],
            this.lsctf.PARAM_orient:cur_lsct_profile_dict[neko_lsct_repeater_template.ORIENTS],
            this.lsctf.PARAM_lsct_type:type
        }
        );
        acfg = this.arm_collate(acfg, name, name, cur_lsct_profile_dict[neko_lsct_repeater_template.DSKCOL_TEMPLATE],True);
        acfg = this.arm_msk_collate(acfg, name, name, cur_lsct_profile_dict[neko_lsct_repeater_template.DSKCOL_TEMPLATE],True);
        return acfg;
    def arm_one_lsct_rand_char(this,acfg,cur_lsct_profile_dict,name,data_root,seed):
        dname=cur_lsct_profile_dict[neko_lsct_char_random_template.DATA_ENDPOINTS][0];
        mname=cur_lsct_profile_dict[neko_lsct_char_random_template.META_ENDPOINTS][0];
        this.lsct_rand_f.arm_agt_core({},{
            this.lsct_rand_f.DATA_PRFX_local:dname,
            this.lsct_rand_f.META_PRFX_local:mname,

        },acfg,P(name,"lsct_rand"),{
            this.lsct_rand_f.PARAM_im_root:cur_lsct_profile_dict[neko_lsct_char_random_template.BG_DB],
            this.lsct_rand_f.PARAM_db_root:cur_lsct_profile_dict[neko_lsct_char_random_template.DB_PATH],
            this.lsct_rand_f.PARAM_seed:seed,
            this.lsct_rand_f.PARAM_K:2,
            this.lsct_rand_f.PARAM_N:64,

            this.lsct_rand_f.PARAM_charset:cur_lsct_profile_dict[neko_lsct_char_random_template.CHARSET],
        }
        );
        acfg = this.arm_collate(acfg, dname, P(dname,"collate"), cur_lsct_profile_dict[neko_lsct_char_random_template.DSKCOL_TEMPLATE],True);
        acfg=this.arm_im_meta_collate(
            acfg, mname, P(mname, "collate"), cur_lsct_profile_dict[neko_lsct_char_random_template.DSKCOL_TEMPLATE], cv2=True
        );
        return acfg;
    def armdict(this,k,seeds):
        return ({P(k,str(seed)): awa_nograd.empty() for seed in seeds},
                {P(k,str(seed)): seed for seed in seeds});

    def bootstrap(this):
        this.seeds_diskldr=this.SEEDS[:this.NUM_diskldr];
        this.seeds_augcol = this.SEEDS[:this.NUM_augcol];
        this.seeds_lsct = this.SEEDS[:this.NUM_lsct];
        this.seeds_meta = this.SEEDS[:this.NUM_meta];
        this.seeds_sifo = this.SEEDS[:this.NUM_sinfo];

    def get_ldr_agent(this,dprofile_dict,mprofile_dict,lsct_profile_dict,pcfg:neko_platform_cfg):

        diskldr_dict,diskldr_scfg=this.armdict("diskldr",this.seeds_diskldr);
        for k in diskldr_scfg:
            this.arm_one_diskldr_agents(diskldr_dict[k], dprofile_dict, pcfg.data_root, seed=diskldr_scfg[k]);

        augcol_dict,augcol_scfg=this.armdict("augcol",this.seeds_augcol);
        for k in augcol_scfg:
            this.arm_one_augcol_agt(augcol_dict[k], dprofile_dict, pcfg.data_root, seed=augcol_dict[k]);

        meta_dict,meta_scfg=this.armdict("meta",this.seeds_meta);
        for k in meta_scfg:
            this.arm_one_training_meta_loader(meta_dict[k], mprofile_dict, pcfg.data_root, seed=meta_scfg[k]);


        sinfo_dict,sinfo_scfg=this.armdict("sinfo",this.seeds_sifo);
        for k in sinfo_scfg:
            this.arm_one_training_sinfo_loader(sinfo_dict[k], mprofile_dict, pcfg.data_root, seed=sinfo_scfg[k]);
        lsct_dict={};
        subd, lsct_scfg = this.armdict("lsct", this.seeds_lsct);

        for pk in lsct_profile_dict:
            cur_lsct_profile=lsct_profile_dict[pk];
            for k in lsct_scfg:
                lsct_dict[k]=subd[k];
                if(cur_lsct_profile["type"]=="lsct_fullrand"):
                    this.arm_one_lsct_rand_char(subd[k],cur_lsct_profile,pk,pcfg.data_root,seed=lsct_scfg[k]);
                else:
                    this.arm_one_shadow_lsct(subd[k],cur_lsct_profile,pk, pcfg.data_root, seed=lsct_scfg[k]);
        return this.LDR_AGT.get_agtcfg(
            diskldr_dict,augcol_dict,meta_dict,sinfo_dict,lsct_dict
        );
    def setup(this):
        super().setup();
        this.bootstrap();
        this.mldrf=this.MLDRF();
        this.im_text_dldrf=this.IM_TEXT_DLDRF();
        # this.panoptic_dldrf = this.PANOPTIC_DLDRF();

        # this.lsctf=this.LSCT_SHAD();
        # this.lsct_rand_f=this.LSCTC_RAND();


    # this will need to wait. We don't fetch here... we fetch when meta is loaded.
    @classmethod
    def fetch_list(cls,dataprfx):
        return [VN.SAM_ID(dataprfx), VN.SAM_ORIG(dataprfx), VN.UTF(dataprfx), VN.GT_TOK_UTF(dataprfx),
         VN.IM_normed_tensor_list(dataprfx), VN.SAM_THUMB_normed_tensor(dataprfx) ] ;



#         pass;
# if __name__ == '__main__':
#     from osocrNG.data_utils.common_data_presets_mk5.data.presets.protocol.train_data_dict_collection import \
#         get_ctw_yolox_tr,get_abi_mjst
#     from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
#     from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
#     import os
#     from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_agent as aa
#     from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
#     CHS_META="meta-chs_3755";
#     EN_META="meta-endigit_uncased";
#     CH_CHAR_DATA="ctwch";
#     EN_WORD_DATA="mjst";
#     LSCT_CH=P(CH_CHAR_DATA,"LSCT_repeat");
#
#     pcfg=neko_platform_cfg(None);
#     dprofile_dict={
#         CH_CHAR_DATA:neko_datastream_profile.get_default_training_im_item_loader(CH_CHAR_DATA,pcfg.data_root,get_ctw_yolox_tr(pcfg.data_root),batchsize=32),
#         EN_WORD_DATA: neko_datastream_profile.get_default_training_im_item_loader(EN_WORD_DATA,pcfg.data_root,get_abi_mjst(pcfg.data_root),batchsize=32)
#
#     };
#     mpath_enchs=os.path.join(pcfg.data_root,"dicts_v3/chs3755_en_uncased_digits/");
#     mpath_en = os.path.join(pcfg.data_root, "dicts_v3/endigit_uncased/");
#
#     mprofile_dict={
#         CHS_META:neko_meta_stream_profile.get_default_training_loader_noto_sample_kshot(
#             CHS_META,mpath_enchs,
#             CH_CHAR_DATA,
#            mpath_enchs, os.path.join(pcfg.data_root,"ctw_ch_yolox/train/")
#         ),
#         EN_META: neko_meta_stream_profile.get_default_training_loader_noto(
#             EN_META,mpath_en,EN_WORD_DATA,mpath_en)
#     };
#     lsct_profile_dict={
#         LSCT_CH:neko_lsct_repeater_template.get_default_lsct_repeater(
#             CH_CHAR_DATA,pcfg.data_root,LSCT_CH)
#     };
#
#
#     acfg=awa.empty();
#     af=neko_32x_training_data_loader_agent_factory(pcfg.data_root);
#     acfg=af.get_ldr_agent(dprofile_dict,mprofile_dict,lsct_profile_dict,pcfg);
#     #
#     # af.arm_one_diskldr_agents(acfg, dprofile_dict,pcfg.data_root,seed=9);
#     # af.arm_one_augcol_agt(acfg,dprofile_dict,pcfg.data_root,seed=9);
#     # af.arm_one_training_meta_loader(acfg,mprofile_dict,pcfg.data_root,9);
#     # af.arm_one_training_sinfo_loader(acfg,mprofile_dict,pcfg.data_root,9);
#     # af.arm_one_shadow_lsct(acfg,lsct_profile_dict,pcfg.data_root,9);
#
#
#     a=af.LDR_AGT.make(acfg);
#     e=neko_environment();
#     for i in tqdm.tqdm(range(10000)):
#         ws = neko_workspace();
#         a.take_action(ws,e);
#     pass;