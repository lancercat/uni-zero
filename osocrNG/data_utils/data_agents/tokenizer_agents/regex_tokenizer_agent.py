from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_regex_based_tokenizer_agent(ama):
    INPUT_text="text";
    OUTPUT_token_utfs="token_utfs";
    PARAM_pattern="pattern";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.text = this.register_input(this.INPUT_text, iocvt_dict);
        this.token_utfs = this.register_output(this.OUTPUT_token_utfs, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.pattern=neko_get_arg(this.PARAM_pattern,params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        text = workspace.get(this.text);
        workspace.add(this.token_utfs,[tokenize(t,this.pattern) for t in text])
        return workspace, environment;
    @classmethod
    def get_agtcfg(cls,
                   text,
                   token_utfs,
                   split_pattern=r"\X"
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_text: text, cls.OUTPUT_token_utfs: token_utfs}, "modcvt_dict": {},cls.PARAM_pattern:split_pattern}};

if __name__ == '__main__':
    neko_regex_based_tokenizer_agent.print_default_setup_scripts()