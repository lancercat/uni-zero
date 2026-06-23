
class neko_abstract_bogo_g2:
    PARAM_MODCVT="mod_cvt"
    def cuda(this):
        print("calling cuda to a bogo module, deprecated and nothing happens");
        pass;
    # Not necessary, unless you want to make the IDE to understand.
    def declare_mods(this):
        pass;

    def forward(this, *args, **kwargs):
        pass;
    def post_setup_hook(this):
        pass;
    def check_cvt(this,mod_cvt_dict):
        return True;
    # Nah it anyway takes more than args as parameter...(wwww)
    def registermods(this,moddict,mod_cvt_dict):
        if(not this.check_cvt(mod_cvt_dict)):
            fatal("error");
        # The callable modules linked
        this.modnames = list(mod_cvt_dict.keys());
        # The actual modules (if you want to have the driver to freeze them)
        this.modinst=[mod_cvt_dict[k] for k in this.modnames];
        for k in mod_cvt_dict:
            this.__setattr__(k,moddict[mod_cvt_dict[k]]);

    def setup(this,args):
        pass;
    def __init__(this, args, moddict):
        # If you want to replicate, take the args away.
        this.args=args;
        this.declare_mods();
        this.setup(this.args);
        mod_cvt_dict=args[this.PARAM_MODCVT];
        this.registermods(moddict,mod_cvt_dict);
        this.post_setup_hook();
        this.save_each=-9


    def __call__(this, *args, **kwargs):
        return this.forward(*args, **kwargs);

    # Fancy controls can yield weird race conditions.
    # The users need to take control of real models behind the bogo modules manually.
    # Gist is, we are not going to sugarcoat the risks with helper APIs in the NG framework
class neko_wrapper_bogo_g2(neko_abstract_bogo_g2):

    @classmethod
    def wrapthis(cls,model):
        return neko_wrapper_bogo_g2({"mod_cvt":{"model":"topnep"}},{"topnep":model});
    def declare_mods(this):
        this.model=None;

    # Nah in NG framework bogomods no more holds control over modules.
    def registermods(this, moddict, mod_cvt_dict):
        for k in mod_cvt_dict:
            this.__setattr__(k, moddict[mod_cvt_dict[k]]);

    def forward(this, *args, **kwargs):
        return this.model(*args,**kwargs);