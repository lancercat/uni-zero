class init_layer_def:
    def __init__(this, layer_dict,bn_dict,container):
        this.conv=container[layer_dict["conv"]];
        this.bn=container[bn_dict["bn"]];
        this.relu=container[layer_dict["relu"]];
    def __call__(this,x,ctrl):
        a=this.conv(x,ctrl);
        b=this.bn(a);
        return this.relu(b);

# the first layer is adjusted by the controller.
class BasicBlock_first_off:
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


    def __call__(this,x,ctrl):
        residual = x
        out = this.conv1(x,ctrl)
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
class BasicBlock_first_nooff:
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


    def __call__(this,x,ctrl):
        residual = x
        out = this.conv1(x,ctrl)
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
# all layers are adjusted by the controller.
class BasicBlock_all_off_ext:
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


    def __call__(this,x,ctrl):
        residual = x
        out = this.conv1(x,ctrl)
        out = this.bn1(out)
        out = this.relu(out)

        out = this.conv2(out,ctrl)
        out = this.bn2(out)

        if this.downsample:
            residual = this.sample_conv(x,ctrl);
            residual = this.sample_bn(residual);
        out += residual
        out = this.relu(out)

        return out
class dan_reslayer_ext_def_1st:
    def __init__(this, layer_dict,bn_dict,container):
        this.blocks=[];
        for k in layer_dict["blocks"]:
            blk=BasicBlock_first_off(layer_dict["blocks"][k],bn_dict["blocks"][k],container)
            this.blocks.append(blk);
    def __call__(this,x,ctrl):
        for i in range(len(this.blocks)):
            x=this.blocks[i](x,ctrl);
        return x;
