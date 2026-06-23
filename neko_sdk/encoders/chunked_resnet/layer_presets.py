from torch import nn

from neko_sdk.chunked_FE.res45.layer_presets import res45_bn as res45_bn_stub
from neko_sdk.chunked_FE.res45.layer_presets import res45_wo_bn as res45_wo_bn_stub


def res45_wo_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True,engine=nn.Conv2d):
    if(blkcnt is None):
        blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    return res45_wo_bn_stub(inpch[0],strides,ochs,blkcnt,inplace=inplace,engine=engine);


def res45_bn(oupch,strides,frac=1,ochs=None,blkcnt=None,affine=True,engine=nn.BatchNorm2d):
    if(blkcnt is None):
        blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    return res45_bn_stub(strides,ochs,blkcnt,affine,engine);


# OSOCR config. Seems they have much better perf due to the heavier layout
# The called the method ``pami'' www


def res45p_wo_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True):
    if(blkcnt is None):
        blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    return res45_wo_bn_stub(inpch[0], strides, ochs, blkcnt, inplace=inplace, engine=nn.Conv2d);


def res45p_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True,affine=True,engine=nn.BatchNorm2d):
    blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    return res45_bn_stub(strides, ochs, blkcnt, affine, engine);
