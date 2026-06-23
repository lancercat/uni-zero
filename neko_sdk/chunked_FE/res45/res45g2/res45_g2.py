from neko_sdk.neko_framework_NG.bogog2_modules.neko_abstract_bogo_g2 import neko_abstract_bogo_g2
from neko_sdk.chunked_FE.res45.res45g2.layers import init_layer_g2,dan_reslayer_g2
class neko_res45_bogo_g2(neko_abstract_bogo_g2):
    INIT_LAYER=init_layer_g2;
    LAYER=dan_reslayer_g2;

    def declare_mods(this):
        this.conv=None;
        this.norm=None;
        this.drop=None;


    def post_setup_hook(this):
        this.init_layer = this.INIT_LAYER(this.conv.name_dict["0"], this.norm.name_dict["0"], this.conv,this.norm);
        this.res_layer1 = this.LAYER(this.conv.name_dict["1"], this.norm.name_dict["1"], this.conv,this.norm);
        this.res_layer2 = this.LAYER(this.conv.name_dict["2"], this.norm.name_dict["2"], this.conv,this.norm);
        this.res_layer3 = this.LAYER(this.conv.name_dict["3"], this.norm.name_dict["3"], this.conv,this.norm);
        this.res_layer4 = this.LAYER(this.conv.name_dict["4"], this.norm.name_dict["4"], this.conv,this.norm);
        this.res_layer5 = this.LAYER(this.conv.name_dict["5"], this.norm.name_dict["5"], this.conv,this.norm);
        # this.debug();

    def debug_core(this,dic,cont):
        for k in dic:
            if(type(dic[k])==dict):
                this.debug_core(dic[k],cont);
            else:
                print(dic[k],":",cont[dic[k]]);
    def debug(this):
        this.debug_core(this.conv.name_dict, this.conv);
        this.debug_core(this.norm.name_dict, this.norm);

    def forward(this, x):
        ret = [];
        x1 = this.init_layer(x.contiguous());
        tmp_shape = x.size()[2:]
        x2 = this.res_layer1(x1);
        if x2.size()[2:] != tmp_shape:
            tmp_shape = x2.size()[2:]
            ret.append(x2)
        x3 = this.res_layer2(x2);
        if x3.size()[2:] != tmp_shape:
            tmp_shape = x3.size()[2:];
            ret.append(x3);
        x4 = this.res_layer3(x3);
        if x4.size()[2:] != tmp_shape:
            tmp_shape = x4.size()[2:];
            ret.append(x4);
        x5 = this.res_layer4(x4);
        if x5.size()[2:] != tmp_shape:
            tmp_shape = x5.size()[2:]
            ret.append(x5);
        x6 = this.res_layer5(x5);
        ret.append(x6);
        return ret

