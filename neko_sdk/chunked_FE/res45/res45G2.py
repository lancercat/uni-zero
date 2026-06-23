# This thing keeps a set of modules (CANNOT be called, you need a bogo module to call it).
from neko_sdk.chunked_FE.res45.layer_presets import res45_wo_bn, res45_bn, res45_ffn_naive, \
    res45_ffn_naive_drop
from neko_sdk.MJT.neko_module_collection import neko_module_collectionNG,neko_module_collection

class neko_r45_layers_orig(neko_module_collection):
    def build_layers(this,**kwargs):
        this.setup_modules(res45_wo_bn(
            inpch=kwargs["inpch"],
            strides=kwargs["strides"],
            ochs=kwargs["ochs"],
            blkcnt=kwargs["blkcnt"],
            inplace=kwargs["inplace"]),
            "root");

class neko_r45_norms_orig(neko_module_collection):
    # inplace ReLUs cannot be trained in parallel.
    def build_layers(this,**kwargs):
        this.setup_modules(res45_bn(
            strides=kwargs["strides"],
            ochs=kwargs["ochs"],
            blkcnt=kwargs["blkcnt"],
            affine=kwargs["affine"]),
            "root");

class neko_r45_ffns_naive(neko_module_collection):
    def build_layers(this,**kwargs):
        this.setup_modules(res45_ffn_naive(
            bochs=kwargs["bochs"],fochs=kwargs["fochs"]),
            "root");
class neko_r45_ffns_naive_drop(neko_module_collection):
    def build_layers(this,**kwargs):
        this.setup_modules(res45_ffn_naive_drop(
            bochs=kwargs["bochs"],fochs=kwargs["fochs"],drop=kwargs["drop"]),
            "root");




class neko_r45_layers_origNG(neko_module_collectionNG):
    PARAM_in_channels="inpch";
    PARAM_strides="strides";
    PARAM_out_channels="ochs";
    PARAM_block_counts="blkcnt";
    PARAM_inplace_relu="inplace";
    PARAM_prefix="prefix";
    def build_layers(this,param):
        this.setup_modules(res45_wo_bn(
            inpch=param[this.PARAM_in_channels],
            strides=param[this.PARAM_strides],
            ochs=param[this.PARAM_out_channels],
            blkcnt=param[this.PARAM_block_counts],
            inplace=param[this.PARAM_inplace_relu]),
            prefix="root");

class neko_r45_norms_origNG(neko_module_collectionNG):
    PARAM_strides="strides";
    PARAM_out_channels="ochs";
    PARAM_block_counts="blkcnt";
    PARAM_affine="affine";
    PARAM_prefix="prefix";
    # inplace ReLUs cannot be trained in parallel.
    # the prefix is only used to index the modules inside the container
    # it can get super ugly should we decompress the modules into the save folder, so we keep them in this "folder".

    def build_layers(this,params):
        this.setup_modules(res45_bn(
            strides=params[this.PARAM_strides],
            ochs=params[this.PARAM_out_channels],
            blkcnt=params[this.PARAM_block_counts],
            affine=params[this.PARAM_affine]),
            "root");

class neko_r45_ffns_naiveNG(neko_module_collectionNG):
    PARAM_ichs="bochs";
    PARAM_ochs="fochs"
    def build_layers(this,params):
        this.setup_modules(res45_ffn_naive(
            bochs=params[this.PARAM_ichs],fochs=params[this.PARAM_ochs]),
            "root");

class neko_r45_ffns_naive_dropNG(neko_module_collectionNG):
    def build_layers(this,params):
        this.setup_modules(res45_ffn_naive_drop(
            bochs=params["bochs"],fochs=params["fochs"],drop=params["drop"]),
            "root");




