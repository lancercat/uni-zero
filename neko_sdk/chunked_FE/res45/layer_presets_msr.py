from torch import nn

from neko_sdk.encoders.chunked_resnet.neko_block_fe import make_body_layer_bn, make_init_layer_bn, \
    make_init_layer_wo_bn, make_body_layer_wo_bn


# Well the details are handed over to configs now.
def res45_wo_bn(inpch,strides,ochs,blkcnt,inplace=True,engine=nn.Conv2d):
    retlayers={};
    retlayers["0"]=make_init_layer_wo_bn(inpch,ochs[0],strides[0],inplace,engine=engine);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1],inplace,engine=engine);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],inplace,engine=engine);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],inplace,engine=engine);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],inplace,engine=engine);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],inplace,engine=engine);
    return retlayers;


def res45_bn(strides,ochs,blkcnt,affine=True,engine=nn.BatchNorm2d):
    retlayers = {};
    retlayers["0"] = make_init_layer_bn(ochs[0],affine=affine,engine=engine);
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1],affine=affine,engine=engine);
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],affine=affine,engine=engine);
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],affine=affine,engine=engine);
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],affine=affine,engine=engine);
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],affine=affine,engine=engine);
    return retlayers;
