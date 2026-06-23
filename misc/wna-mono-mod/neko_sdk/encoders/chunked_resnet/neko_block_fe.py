from torch import nn


def conv1x1(in_planes,out_planes,stride=1,engine=nn.Conv2d):
    return engine(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)
def conv3x3(in_planes, out_planes, stride=1,engine=nn.Conv2d):
    "3x3 convolution with padding"
    return engine(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class norm_dict:
    def __init__(this):
        this.dic={"bn":nn.BatchNorm2d,
                  "in":nn.InstanceNorm2d}
    def __call__(this, type):
        return this.dic[type];

def make_init_layer_cco_wo_bn(inpch,outplanes,stride,engine=nn.Conv2d):
    conv = engine(inpch, outplanes, kernel_size=3, stride=stride, padding=1,
                           bias=False)
    relu = nn.ReLU(inplace=True)
    return {"conv":conv,"relu":relu};

def make_init_layer_wo_bn(inpch,outplanes,stride,inplace=True,engine=nn.Conv2d):
    conv = engine(inpch, outplanes, kernel_size=3, stride=stride, padding=1,
                           bias=False)
    relu = nn.ReLU(inplace=inplace)
    return {"conv":conv,"relu":relu};

def make_fin_layer_wo_bn(inpch,outplanes,stride,inplace=True,engine=nn.Conv2d,act_engine=None):
    conv = engine(inpch, outplanes, kernel_size=1, stride=stride, padding=0,
                           bias=False);
    return {"conv":conv};



def make_init_layer_bn(outplanes,affine=True,engine=nn.BatchNorm2d):
    bn = engine(outplanes,affine=affine)
    return {"bn":bn};

def make_block_wo_bn(inplanes, outplanes, stride=1,inplace=True,engine=nn.Conv2d):
    return{
        "conv1" :conv1x1(inplanes, outplanes,engine=engine),
        "relu" : nn.ReLU(inplace=inplace),
        "conv2" : conv3x3(outplanes, outplanes, stride,engine=engine),
   }
def make_block_bn(outplanes,affine=True,engine=nn.BatchNorm2d):
    return{
        "bn1": engine(outplanes,affine=affine),
        "bn2": engine(outplanes,affine=affine),
    }


def make_dowsample_layer_bn( expansion, planes,affine=True,engine=nn.BatchNorm2d):
    return {"bn":engine(planes * expansion,affine=affine)};
def make_dowsample_layer( inplanes, expansion, planes, stride=1,engine=nn.Conv2d):
    return {"conv":engine(inplanes, planes * expansion,
                      kernel_size=1, stride=stride, bias=False)};

# we only decouple at layer level---- Or it can get seriously messy.
def make_body_layer_wo_bn( inplanes,blocks, planes, expansion, stride=1,inplace=True,engine=nn.Conv2d):
    ret_weight={};
    ret_weight["blocks"]={};
    ret_weight["blocks"]["0"]=make_block_wo_bn(inplanes,planes,stride,inplace,engine=engine);
    if stride != 1 or inplanes != planes * expansion:
        ret_weight["blocks"]["0"]["downsample"]=make_dowsample_layer(inplanes,expansion, planes,stride,engine=engine)
    for i in range(1, blocks):
        ret_weight["blocks"][str(i)]=make_block_wo_bn(planes,planes,inplace=inplace,engine=engine)
    return ret_weight;

def make_body_layer_bn( inplanes,blocks, planes, expansion, stride=1,affine=True,engine=nn.BatchNorm2d):
    ret_weight={};
    ret_weight["blocks"]={};
    ret_weight["blocks"]["0"]=make_block_bn(planes,affine=affine,engine=engine);
    if stride != 1 or inplanes != planes * expansion:
        ret_weight["blocks"]["0"]["downsample"]=make_dowsample_layer_bn(expansion,planes,affine=affine,engine=engine)
    for i in range(1, blocks):
        ret_weight["blocks"][str(i)]=make_block_bn(planes,affine=affine,engine=engine)
    return ret_weight;

class init_layer:
    def __init__(this, layer_dict,bn_dict,container):
        this.conv=container[layer_dict["conv"]];
        this.bn=container[bn_dict["bn"]];
        this.relu=container[layer_dict["relu"]];
    def __call__(this,x):
        a=this.conv(x);
        b=this.bn(a);
        return this.relu(b);

# assembles from dicts.
# Generally this module does onw the modules---which means we do not save or load via it
class BasicBlock_ass:
    def __init__(this, layer_dict,bn_dict,container):
        this.conv1=container[layer_dict["conv1"]];
        this.relu=container[layer_dict["relu"]];
        this.bn1=container[bn_dict["bn1"]];
        this.conv2=container[layer_dict["conv2"]];
        this.bn2 = container[bn_dict["bn2"]];
        this.downsample =False;
        if("downsample" in layer_dict):
            this.sample_conv=container[layer_dict["downsample"]["conv"]];
            this.sample_bn=container[bn_dict["downsample"]["bn"]];
            this.downsample = True;


    def __call__(this,x ):
        residual = x
        out = this.conv1(x)
        out = this.bn1(out)
        out = this.relu(out)

        out = this.conv2(out)
        out = this.bn2(out)

        if this.downsample:
            residual = this.sample_conv(x);
            residual = this.sample_bn(residual);
        out += residual
        out = this.relu(out)

        return out


class dan_reslayer:
    def __init__(this, layer_dict,bn_dict,container):
        this.blocks=[];
        for k in layer_dict["blocks"]:
            blk=BasicBlock_ass(layer_dict["blocks"][k],bn_dict["blocks"][k],container)
            this.blocks.append(blk);
    def __call__(this,x):
        for l in this.blocks:
            x=l(x);
        return x;
