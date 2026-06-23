import os.path

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_abstract_saving_agent(neko_module_wrapping_agent):
    PARAM_save_path = "save_path";
    PARAM_prefix = "img_prefix";
    def set_etc(this,param):
        this.export_root=neko_get_arg(this.PARAM_save_path,param);
        this.prefix=neko_get_arg(this.PARAM_prefix,param);

        this.cntr=0;
        this.testname="nep";

    def report(this,*param):
        pass;
    def reset(this,test_name,hasunk):
        this.cntr = 0;
        this.testname=test_name;
        this.export_path=os.path.join(this.export_root,this.testname);
        os.makedirs(this.export_path,exist_ok=True);
