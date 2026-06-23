from torch import nn
from neko_sdk.log import fatal
# Wraps a huge collection of things
class neko_module_collection(nn.Module):
    def __init__(this,**kwargs):
        super(neko_module_collection, this).__init__()
        this.build_layers(**kwargs);
    def __getitem__(this, item):
        return this._modules[item];
    def build_layers(this,**kwargs):
        pass;
    def setup_modules_core(this,mdict,prefix):
        name_dict={};
        for k in mdict:
            if (type(mdict[k]) is dict):
                subname_dict = this.setup_modules_core(mdict[k], prefix + "_" + k);
                name_dict[k] = subname_dict;
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                name_dict[k] = prefix + "_" + k;
        return name_dict;
    def setup_modules(this,mdict,prefix):
        this.name_dict=this.setup_modules_core(mdict,prefix);

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        fatal("error");


# Wraps a huge collection of things
class neko_module_collectionNG(nn.Module):
    def __init__(this,param):
        super(neko_module_collectionNG, this).__init__()
        this.build_layers(param);
    def __getitem__(this, item):
        return this._modules[item];
    def build_layers(this,param):
        pass;
    def setup_modules_core(this,mdict,prefix):
        name_dict={};
        for k in mdict:
            if (type(mdict[k]) is dict):
                subname_dict = this.setup_modules_core(mdict[k], prefix + "_" + k);
                name_dict[k] = subname_dict;
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                name_dict[k] = prefix + "_" + k;
        return name_dict;
    def setup_modules(this,mdict,prefix):
        this.name_dict=this.setup_modules_core(mdict,prefix);

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        fatal("error");