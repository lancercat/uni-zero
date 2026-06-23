from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

# so its abstract now. The point is that now it can go with meta module,
# which will be handy bcs tokenization is closely tied to meta design.
# the downside is there are 2 tokenization--- one at meta sampling, and another at loss computations
# the upside is now we can have different tokenization in one model.
# given string operation is actuall a few ms so we bite the bullet here
# and when we have meta factories the will be routed properly
class neko_module_based_tokenizer_agent(ama):
    INPUT_text="text";
    OUTPUT_token_utfs="token_utfs";
    MOD_tokenize_mod="tokenize_mod";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.text = this.register_input(this.INPUT_text, iocvt_dict);
        this.token_utfs = this.register_output(this.OUTPUT_token_utfs, iocvt_dict);
        this.tokenize_mod = this.register_mod(this.MOD_tokenize_mod, modcvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        text = workspace.get(this.text);
        workspace.add(this.token_utfs,environment(this.tokenize_mod(text)))
        return workspace, environment;
    @classmethod
    def get_agtcfg(cls,
                   text,
                   token_utfs,
                   tokenize_mod
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_text: text, cls.OUTPUT_token_utfs: token_utfs},
                                         "modcvt_dict": {cls.MOD_tokenize_mod: tokenize_mod}}}

if __name__ == '__main__':
    neko_module_based_tokenizer_agent.print_default_setup_scripts()