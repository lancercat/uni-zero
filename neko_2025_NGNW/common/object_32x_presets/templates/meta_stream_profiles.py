import os.path

from neko_sdk.neko_framework_NG.names import P
from neko_2025_NGNW.common.object_32x_presets.templates.dataspec import neko_basic_im_dataspec
class neko_meta_repo:
    PATH="path";
    @classmethod
    def get_default_meta_repo(cls,path):
        return {
            cls.PATH:path
        };
        pass;
# this defines how the meta is going to be sampled
class neko_meta_sampler:
    META_prfx="meta_prfx"; # composition instea
    CAPACITY="capacity";
    FRAC = "frac";
    REF_DATA_prfx_list="REF_DATA_prfx_list"
    @classmethod
    def get_dft_sampled_name(cls,prfx):
        return P(prfx,"sampled");

    @classmethod
    def get_sampled_meta_repo(cls,metaprfx,refdataprfx,frac=0.8,capacity=128):
        return {
            cls.META_prfx:metaprfx,
            cls.CAPACITY:capacity,
            cls.FRAC:frac,
            cls.REF_DATA_prfx_list:[refdataprfx]
        };

class neko_sideinfo_repo:
    TYPE_GLYPH_PT="glyphpt"; # one single pt, can be loaded into
    TYPE_IM_DB="imdb"; # an lmdb of support samples. we are not differentiating glyph db from image db. Gist is that you can figure out yourself.
    TYPE_SEQ_DB="seq_db"; # sequences, can be dict definition, radicals, strokes, or whatever
    TYPE_LONG_ID="long_id"; # no sideinfo for ye. the network part will figure based on utf and tdict.
    TYPE_ID="ids"; # just ids--- u can pass this to an embedding layer to get ol-good embedding based class centers.

    FORCE_GREY= "force_grey"; # this is to avoid mass storage waste if we want to see if color play a role.
    TYPE="type";
    DBCFG = "main_db_path";
    TEMPLATE="template";

    NAME="name";

    # even if you want to use synthetic char image by combining glyph db with some rnd bgs as samples or protos,
    # do it in lsct, not here.

    K = "k";
    REPLACEMENT="replacement"; # replacement  during sampling, will lead to multiple identical sideinfo for one class, if k>1


    @classmethod
    def get_dft_id_name(cls, prfx):
        return P(prfx, "longid");


    @classmethod
    def get_dft_noto_name(cls, prfx):
        return P(prfx, "noto");

    @classmethod
    def get_dft_imshot_name(cls, prfx):
        return P(prfx, "imshot");

    @classmethod
    def get_default_longid(cls,name,path):
        return {
            neko_sideinfo_repo.TYPE: neko_sideinfo_repo.TYPE_LONG_ID,
            neko_sideinfo_repo.NAME: name,
            neko_sideinfo_repo.DBCFG: path,
            # well we don't call it utf bcs u may have a dict of smth.
            # none is just saying using key itself (to index embeddings or smth)
        }

    @classmethod
    def get_default_glyphpt(cls,name,path,template):
        return {
            neko_sideinfo_repo.TYPE: neko_sideinfo_repo.TYPE_GLYPH_PT,
            neko_sideinfo_repo.DBCFG: path,
            neko_sideinfo_repo.TEMPLATE: template,
            neko_sideinfo_repo.NAME: name
        }
    @classmethod
    def get_default_imdb(cls,name,root_dict,template,k,force_grey_scale=False):
        return {
            neko_sideinfo_repo.TYPE: neko_sideinfo_repo.TYPE_IM_DB,
            neko_sideinfo_repo.DBCFG: root_dict,
            neko_sideinfo_repo.TEMPLATE: template,
            neko_sideinfo_repo.K: k,
            neko_sideinfo_repo.FORCE_GREY:force_grey_scale,
            neko_sideinfo_repo.NAME: name
        }
class neko_im_meta_template(neko_basic_im_dataspec):
    @classmethod
    def get_default_im_meta_template(cls,sz=32):
        return cls.get_default_template(sz,"meta");


class neko_meta_stream_profile:

    META_REPO="meta_repo";
    META_SAMPLER="meta_sampler";
    SIDE_INFO_REPOS="side_infos";
    SIDE_INFO_MBND="side_info_modprfx"; # module binding
    META_ENDPOINTS="meta_endpoints";
    ENDPOINTS="endpoints"; # note we don't arm them here
    FE_MOD_TIED="mod_tied";  # module tied elsewhere, if true will not spawn own weights.
    # More on arming.
    # # ## eventual layout on label space [CENTERED_SPTOKENS, SINFO_TOKENS, UNCENTERED_SPTOKENS]
    # This loader only loads the SINFO_TOKENS;
    # # These are of some patterns that are not described in sideinfo
    # # say [s]/[BOS]/[EOS]/[blank], etc, these goes on lower end
    # UNCENTERED_SPTOKS="sptok_non_centered";
    # # these are not of a pattern.  [UNK]. those goes on higher end
    # # bcs these are two different type of things, so they reside on two sides of the label space.

    # we list these here, however arming sptoks are NOT the job of meta_stream for diskldr. It's defined by tasks.


    @classmethod
    def arm_default_sinfo_kshot(cls,cfg,im_shot_root,imshot_modbinding,imshot_k,prfx,imshot_name=None,im_shot_template=None,mod_tied=False,force_grey=False):
        if(imshot_name is None):
            imshot_name=neko_sideinfo_repo.get_dft_imshot_name(prfx); # this will be the dataname
        if(im_shot_template is None):
            im_shot_template=neko_im_meta_template.get_default_im_meta_template();
        cfg[cls.SIDE_INFO_REPOS][imshot_name]= neko_sideinfo_repo.get_default_imdb(
            imshot_name, {"main": im_shot_root}, im_shot_template,imshot_k,force_grey);  # if you need more than one pass a real dict.
        cfg[cls.ENDPOINTS].append(imshot_name);
        cfg[cls.SIDE_INFO_MBND][imshot_name] = imshot_modbinding;
        cfg[cls.FE_MOD_TIED]=mod_tied;

        return cfg;

    @classmethod
    def arm_default_sinfo_noto(cls, cfg, noto_root,sinfo_mod_prfx_binding, prfx, noto_name=None, noto_template=None):
        if(noto_name is None):
            noto_name=neko_sideinfo_repo.get_dft_noto_name(prfx);
        if(noto_template is None):
            noto_template=neko_im_meta_template.get_default_im_meta_template();
        cfg[cls.SIDE_INFO_REPOS][noto_name] =neko_sideinfo_repo.get_default_glyphpt(
            noto_name, noto_root, noto_template);
        cfg[cls.ENDPOINTS].append(noto_name);
        cfg[cls.SIDE_INFO_MBND][noto_name] = sinfo_mod_prfx_binding;
        cfg[cls.FE_MOD_TIED]=False;

        return cfg;

    @classmethod
    def arm_default_sinfo_longid(cls, cfg,meta_root,sinfo_mod_prfx_binding, prfx, sinfo_name=None):
        if(sinfo_name is None):
            sinfo_name=neko_sideinfo_repo.get_dft_id_name(prfx);
        cfg[cls.SIDE_INFO_REPOS][sinfo_name] =neko_sideinfo_repo.get_default_longid(sinfo_name,meta_root);
        cfg[cls.ENDPOINTS].append(sinfo_name);
        cfg[cls.SIDE_INFO_MBND][sinfo_name] = sinfo_mod_prfx_binding;
        cfg[cls.FE_MOD_TIED]=False;

        return cfg;

    @classmethod
    def get_default_empty(
            cls, meta_name, meta_path,
    ):
        cdict = {
            cls.META_REPO: neko_meta_repo.get_default_meta_repo(meta_path),
            cls.SIDE_INFO_REPOS: {},
            cls.SIDE_INFO_MBND:{}, # signature is a string indicating its modality and domain (script, language, or some group of them).
            cls.ENDPOINTS: [],
            cls.META_ENDPOINTS:[meta_name],
            cls.FE_MOD_TIED:False
        };
        return cdict,meta_name;

    @classmethod
    def get_default_empty_sampled(
            cls,meta_name,meta_path,
            refdata_prfx,
            sampled_name=None,frac=0.8,capacity=128,
    ):
        cdict,meta_name=cls.get_default_empty(meta_name,meta_path);
        if(sampled_name is None):
            sampled_name=neko_meta_sampler.get_dft_sampled_name(meta_name);
        cdict[cls.META_SAMPLER]= neko_meta_sampler.get_sampled_meta_repo(meta_name, refdata_prfx, frac, capacity);
        return cdict,sampled_name

        # noto-2shot sample.
    # its 2-shot to reduce mem footprint
    @classmethod
    def get_default_training_loader_noto_sample_kshot(
            cls,meta_name,meta_path,
            refdata_prfx,
            noto_root,noto_sig,
            im_shot_root,imshot_sig,
            sampled_name=None,frac=0.8,capacity=128,
            noto_name=None,noto_template=None,
            imshot_name=None,imshot_k=2,im_shot_template=None
    ):
        cdict,prfx=cls.get_default_empty_sampled(meta_name,meta_path,refdata_prfx,sampled_name,frac,capacity);
        cdict=cls.arm_default_sinfo_kshot(cdict,im_shot_root,imshot_sig,imshot_k,prfx,imshot_name,im_shot_template);
        cdict=cls.arm_default_sinfo_noto(cdict,noto_root,noto_sig,prfx,noto_name,noto_template);
        return cdict;
    @classmethod
    def get_default_training_loader_sample_kshot(
            cls,meta_name,meta_path,
            refdata_prfx,
            im_shot_root,imshot_sig,
            sampled_name=None,frac=0.8,capacity=128,
            imshot_name=None,imshot_k=2,im_shot_template=None,mod_tied=False,force_grey=False
    ):
        cdict,prfx=cls.get_default_empty_sampled(meta_name,meta_path,refdata_prfx,sampled_name,frac,capacity);
        cdict=cls.arm_default_sinfo_kshot(cdict,im_shot_root,imshot_sig,imshot_k,prfx,imshot_name,im_shot_template,mod_tied=mod_tied,force_grey=force_grey);
        return cdict;
    @classmethod
    def get_default_training_loader_noto(
        cls,meta_name,meta_path,
            refdata_prfx,
            noto_root,noto_sig,
            sampled_name=None,frac=0.8,capacity=128,
            noto_name=None,noto_template=None
    ):
        cdict, prfx = cls.get_default_empty_sampled(meta_name, meta_path, refdata_prfx, sampled_name, frac, capacity);
        cdict = cls.arm_default_sinfo_noto(cdict, noto_root,noto_sig, prfx, noto_name, noto_template);
        return cdict;

    # sinfo is a utf string --- i would not recommend this to be abused into the radical/stroke seq but eh whatsoever
    @classmethod
    def get_default_training_loader_longid(
            cls, meta_name, meta_path,
            refdata_prfx,
            sampled_name=None, frac=0.8, capacity=128,
            id_name=None
    ):
        # a meta defines atomic sub class primary keys (utf string) and their mapping
        cdict, prfx = cls.get_default_empty_sampled(meta_name, meta_path, refdata_prfx, sampled_name, frac, capacity);
        # a sideinformation  defines each subclass er
        cdict = cls.arm_default_sinfo_longid(cdict,meta_path,prfx , prfx,id_name);

        return cdict;

    @classmethod
    def get_default_test_loader_noto(
        cls,meta_name,meta_path,
            noto_root,noto_sig,
            noto_name=None,noto_template=None
    ):
        cdict, prfx = cls.get_default_empty(meta_name, meta_path);
        cdict = cls.arm_default_sinfo_noto(cdict, noto_root,noto_sig, prfx, noto_name, noto_template);
        return cdict;

    @classmethod
    def get_default_testing_loader_sample_kshot(
            cls,meta_name,meta_path,
            im_shot_root,imshot_sig,
            imshot_name=None,imshot_k=2,im_shot_template=None,mod_tied=False,force_grey=False
    ):
        cdict, prfx = cls.get_default_empty(meta_name, meta_path);
        cdict=cls.arm_default_sinfo_kshot(
            cdict,im_shot_root,imshot_sig,imshot_k,prfx,imshot_name,im_shot_template,mod_tied,force_grey);
        return cdict;


# loader does not arm meta.