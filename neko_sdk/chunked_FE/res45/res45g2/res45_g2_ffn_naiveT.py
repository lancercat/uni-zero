from neko_sdk.neko_framework_NG.bogog2_modules.neko_abstract_bogo_g2 import neko_abstract_bogo_g2
from neko_sdk.chunked_FE.res45.res45g2.layers import init_layer_g2,dan_reslayer_g2
# This is a temporary module.
# We will soon deprecate the module after reach milestone of OSOCR-NG.
# There are more than one ugly things, but bear with it till we reach the other end.
class neko_res45_bogo_g2_ffnT(neko_abstract_bogo_g2):

    def declare_mods(this):
        this.conv=None;
        this.norm=None;
        this.drop=None;
        this.ffn=None;

    def post_setup_hook(this):
        this.init_layer = init_layer_g2(this.conv.name_dict["0"], this.norm.name_dict["0"], this.conv,this.norm);
        this.res_layer1 = dan_reslayer_g2(this.conv.name_dict["1"], this.norm.name_dict["1"], this.conv,this.norm);
        this.res_layer2 = dan_reslayer_g2(this.conv.name_dict["2"], this.norm.name_dict["2"], this.conv,this.norm);
        this.res_layer3 = dan_reslayer_g2(this.conv.name_dict["3"], this.norm.name_dict["3"], this.conv,this.norm);
        this.res_layer4 = dan_reslayer_g2(this.conv.name_dict["4"], this.norm.name_dict["4"], this.conv,this.norm);
        this.res_layer5 = dan_reslayer_g2(this.conv.name_dict["5"], this.norm.name_dict["5"], this.conv,this.norm);
        this.ffn_layer1 = this.ffn[this.ffn.name_dict["1"]];
        this.ffn_layer2 = this.ffn[this.ffn.name_dict["2"]];
        this.ffn_layer3 = this.ffn[this.ffn.name_dict["3"]];
        this.ffn_layer4 = this.ffn[this.ffn.name_dict["4"]];
        this.ffn_layer5 = this.ffn[this.ffn.name_dict["5"]];


    def forward(this, x):
        fret = [];
        aret=[];
        x1 = this.init_layer(x.contiguous());
        tmp_shape = x.size()[2:]
        x2 = this.res_layer1(x1);
        if x2.size()[2:] != tmp_shape:
            tmp_shape = x2.size()[2:]
            aret.append(x2);
            fret.append(this.ffn_layer1(x2))
        x3 = this.res_layer2(x2);
        if x3.size()[2:] != tmp_shape:
            tmp_shape = x3.size()[2:];
            aret.append(x3);
            fret.append(this.ffn_layer2(x3));
        x4 = this.res_layer3(x3);
        if x4.size()[2:] != tmp_shape:
            tmp_shape = x4.size()[2:];
            aret.append(x4);
            fret.append(this.ffn_layer3(x4));
        x5 = this.res_layer4(x4);
        if x5.size()[2:] != tmp_shape:
            tmp_shape = x5.size()[2:]
            aret.append(x5);
            fret.append(this.ffn_layer4(x5));
        x6 = this.res_layer5(x5);
        aret.append(x6);
        fret.append(this.ffn_layer5(x6));
        return fret,aret;
