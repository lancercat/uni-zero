# This thing keeps a set of modules (CANNOT be called, you need a bogo module to call it).
from neko_sdk.chunked_FE.res45.res45g3.layer_presets import res45g3_wo_bn, res45g3_bn
from neko_sdk.MJT.neko_module_collection import neko_module_collectionNG

class neko_r45g3_layers_origNG(neko_module_collectionNG):
    def build_layers(this,param):
        this.setup_modules(res45g3_wo_bn(
            inpch=param["inpch"],
            strides=param["strides"],
            ochs=param["ochs"],
            blkcnt=param["blkcnt"],
            inplace=param["inplace"]),
            "root");

class neko_r45g3_norms_origNG(neko_module_collectionNG):
    # inplace ReLUs cannot be trained in parallel.
    def build_layers(this, params):
        this.setup_modules(res45g3_bn(
            strides=params["strides"],
            ochs=params["ochs"],
            blkcnt=params["blkcnt"],
            affine=params["affine"]),
            "root");
