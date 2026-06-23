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
    def build_layers(this,param):
        this.setup_modules(res45_wo_bn(
            inpch=param["inpch"],
            strides=param["strides"],
            ochs=param["ochs"],
            blkcnt=param["blkcnt"],
            inplace=param["inplace"]),
            "root");

class neko_r45_norms_origNG(neko_module_collectionNG):
    # inplace ReLUs cannot be trained in parallel.
    def build_layers(this,params):
        this.setup_modules(res45_bn(
            strides=params["strides"],
            ochs=params["ochs"],
            blkcnt=params["blkcnt"],
            affine=params["affine"]),
            "root");

class neko_r45_ffns_naiveNG(neko_module_collectionNG):
    def build_layers(this,**kwargs):
        this.setup_modules(res45_ffn_naive(
            bochs=kwargs["bochs"],fochs=kwargs["fochs"]),
            "root");
class neko_r45_ffns_naive_dropNG(neko_module_collectionNG):
    def build_layers(this,**kwargs):
        this.setup_modules(res45_ffn_naive_drop(
            bochs=kwargs["bochs"],fochs=kwargs["fochs"],drop=kwargs["drop"]),
            "root");




