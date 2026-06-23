class init_layer_g2:
    def __init__(this, layer_dict,bn_dict,layer_container,bn_container):
        this.conv=layer_container[layer_dict["conv"]];
        this.bn=bn_container[bn_dict["bn"]];
        this.relu=layer_container[layer_dict["relu"]];
    def __call__(this,x):
        a=this.conv(x);
        b=this.bn(a);
        return this.relu(b);


class ffn_layer_g2_normed:
    def __init__(this,layer_dict,bn_dict,layer_container,bn_container):
        this.conv=layer_container[layer_dict["conv"]];
        this.bn=bn_container[bn_dict["bn"]];

    def __call__(this, x):
        a = this.conv(x);
        return this.bn(a);

class ffn_layer_g2_unnormed:
    def __init__(this,layer_dict,bn_dict,layer_container,bn_container):
        this.conv=layer_container[layer_dict["conv"]];

    def __call__(this, x):
        return this.conv(x);


class BasicBlock_ass_g2:
    def __init__(this,  layer_dict,bn_dict,layer_container,bn_container):
        this.conv1=layer_container[layer_dict["conv1"]];
        this.relu=layer_container[layer_dict["relu"]];
        this.bn1=bn_container[bn_dict["bn1"]];
        this.conv2=layer_container[layer_dict["conv2"]];
        this.bn2 = bn_container[bn_dict["bn2"]];
        this.downsample =False;
        if("downsample" in layer_dict):
            this.sample_conv=layer_container[layer_dict["downsample"]["conv"]];
            this.sample_bn=bn_container[bn_dict["downsample"]["bn"]];
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


class dan_reslayer_g2:
    BLK=BasicBlock_ass_g2
    def __init__(this,  layer_dict,bn_dict,layer_container,bn_container):
        this.blocks=[];
        for k in layer_dict["blocks"]:
            blk=this.BLK(layer_dict["blocks"][k],bn_dict["blocks"][k],layer_container,bn_container)
            this.blocks.append(blk);
    def __call__(this,x):
        for l in this.blocks:
            x=l(x);
        return x;

class BasicBlock_ass_g2_nf_na_last(BasicBlock_ass_g2):
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
        return out

class dan_nf_na_last_reslayer_g2(dan_reslayer_g2):
    LAST_BLK=BasicBlock_ass_g2_nf_na_last
    BLK=BasicBlock_ass_g2
    def __init__(this,  layer_dict,bn_dict,layer_container,bn_container):
        this.blocks=[];
        for kid in range(len(layer_dict["blocks"])-1):
            k=str(kid)
            blk=this.BLK(layer_dict["blocks"][k],bn_dict["blocks"][k],layer_container,bn_container)
            this.blocks.append(blk);
        k=str(len(layer_dict["blocks"])-1);
        fblk = this.LAST_BLK(layer_dict["blocks"][k], bn_dict["blocks"][k], layer_container, bn_container)
        this.blocks.append(fblk);


class BasicBlock_ass_mask_g2(BasicBlock_ass_g2):
    def __call__(this,x,mask):
        residual = x
        out,mask = this.conv1(x,mask)
        out,mask = this.bn1(out,mask)
        out = this.relu(out)

        out,mask = this.conv2(out,mask)
        out,mask = this.bn2(out,mask)

        if this.downsample:
            residual,mask = this.sample_conv(x,mask);
            residual,mask = this.sample_bn(residual,mask);
        out += residual;
        out = this.relu(out);
        return out,mask

class dan_reslayer_mask_g2(dan_reslayer_g2):
    BLK=BasicBlock_ass_mask_g2;
    def __call__(this,x,mask):
        for l in this.blocks:
            x,mask=l(x,mask);
        return x,mask;