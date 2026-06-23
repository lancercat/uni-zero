from neko_sdk.chunked_FE.res45 import neko_binorm_common
# so this thing keeps the modules and

from neko_sdk.encoders.chunked_resnet.layer_presets import  make_init_layer_bn,make_body_layer_bn,res45_wo_bn
def res45ss_bn_skip(strides,ochs,blkcnt=None,inplace=True,affine=True,skip=None):
    blkcnt = [None, 3, 4, 6, 6, 3];
    retlayers = {};
    if(skip is None):
        skip=set();
    if(0 not in skip):
        retlayers["0"] = make_init_layer_bn(ochs[0], affine=affine);
    if(1 not in skip):
        retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1],affine=affine);
    if (2 not in skip):
        retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],affine=affine);
    if (3 not in skip):
        retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],affine=affine);
    if (4 not in skip):
        retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],affine=affine);
    if (5 not in skip):
        retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],affine=affine);
    return retlayers;
def res45ss_bn_specific(strides,ochs,blkcnt=None,inplace=True,affine=True,skip=None):
    blkcnt = [None, 3, 4, 6, 6, 3];
    retlayers = {};
    if(skip is None):
        skip=set();
    if(0 in skip):
        retlayers["0"] = make_init_layer_bn(ochs[0], affine=affine);
    if(1  in skip):
        retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1],affine=affine);
    if (2 in skip):
        retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],affine=affine);
    if (3 in skip):
        retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],affine=affine);
    if (4 in skip):
        retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],affine=affine);
    if (5 in skip):
        retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],affine=affine);
    return retlayers;

class neko_r45_binorm_orig_ss(neko_binorm_common):
    # inplace ReLUs cannot be trained in parallel.
    def get_ochs(this,frac,oupch):
        return [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]

    def merge_bn_dicts(this,mdict,sdict,bn_name):
        ret_dict={};
        for k in sdict:
            ret_dict[k]=sdict[k];
        for k in mdict:
            ret_dict[k]=mdict[k];
        return ret_dict;

    def __init__(this, strides, compress_layer, input_shape, bogo_names, bn_names, hardness=2, oupch=512, expf=1,
                 ochs=None, inplace=False, bn_affine=True, specific=2):
        super(neko_r45_binorm_orig_ss, this).__init__();
        if (strides is None):
            strides = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)];
        this.bogo_modules = {};
        layers = res45_wo_bn(input_shape, oupch, strides, frac=expf, inplace=inplace,ochs=this.get_ochs(frac=expf,oupch=oupch));
        this.layer_names = this.setup_modules(layers, "shared_fe");

        this.bogo_names = bogo_names;
        this.bns = [];
        this.named_bn_dicts = {};
        this.named_bn_name_grps = {};
        this.bn_names = bn_names;
        this.specific=specific;

        bns = res45ss_bn_skip( strides ,this.get_ochs(frac=expf,oupch=oupch), affine=bn_affine,skip=specific);
        this.named_bn_name_grps["shared_bn"] = this.setup_bn_modules(bns, "shared_bn", "shared_bn");

        for i in range(len(bogo_names)):
            bn_name = bn_names[i];
            bns = res45ss_bn_specific(strides ,this.get_ochs(frac=expf,oupch=oupch), affine=bn_affine, skip=specific);
            sbn=this.setup_bn_modules(bns, bn_name, bn_name)
            this.named_bn_name_grps[bn_name] = this.merge_bn_dicts(sbn,this.named_bn_name_grps["shared_bn"],bn_name);

        this.bogo_modules = this.refresh_bogo();

class neko_r45p_binorm_orig_ss(neko_r45_binorm_orig_ss):
    # inplace ReLUs cannot be trained in parallel.
    def get_ochs(this,frac,oupch):
        return  [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]

