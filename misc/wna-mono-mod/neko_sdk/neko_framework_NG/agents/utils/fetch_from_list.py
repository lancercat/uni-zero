from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_fetch_from_list(neko_module_wrapping_agent):
    INPUT_item_list="item_list";
    OUTPUT_item="item";
    PARAM_index="index";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.item_list = this.register_input(this.INPUT_item_list, iocvt_dict);
        this.item = this.register_output(this.OUTPUT_item, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.index = neko_get_arg(this.PARAM_index, params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        item_list = workspace.get(this.item_list);
        workspace.add(this.item,item_list[this.index]);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   item_list,
                   item,
                   index
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_item_list: item_list, cls.OUTPUT_item: item},
                                         cls.PARAM_index: index, "modcvt_dict": {}}}

    def get_neko_fetch_from_list(
            item_list,
            item,
            index
    ):
        engine = neko_fetch_from_list;
        return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_item_list: item_list, engine.OUTPUT_item: item},
                                            engine.PARAM_index: index, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_fetch_from_list.print_default_setup_scripts()