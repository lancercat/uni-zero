from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.sptokens import tUNK, tDC, tSPLIT_OCR
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama

# unk is now built independently before loss and/or reward after tokenize.

class neko_token_unk_agent(ama):
    INPUT_token_utfs="token_utfs";
    INPUT_tdict="tdict";
    OUTPUT_token_with_unk="token_with_unk";
    OUTPUT_text_with_unk = "text_with_unk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.token_utfs = this.register_input(this.INPUT_token_utfs, iocvt_dict);
        this.text_with_unk = this.register_output(this.OUTPUT_text_with_unk, iocvt_dict);
        this.token_with_unk = this.register_output(this.OUTPUT_token_with_unk, iocvt_dict);
        pass;

    def set_etc(this, params):
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        raw_utf_token = workspace.get(this.token_utfs);
        utf_with_unk=[];
        tunkrep=tdict[tdict[tUNK]];
        text_wunks=[];
        for samtok in raw_utf_token:
            ul=[t if t in tdict else tUNK for t in samtok];
            textwunk="".join([t if t in tdict else tunkrep for t in samtok]);
            utf_with_unk.append(ul);
            text_wunks.append(textwunk);
        workspace.add(this.token_with_unk,utf_with_unk);
        workspace.add(this.text_with_unk,text_wunks);

        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tdict, token_utfs,
                   text_with_unk, token_with_unk
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_tdict: tdict, cls.INPUT_token_utfs: token_utfs,
                                                        cls.OUTPUT_text_with_unk: text_with_unk,
                                                        cls.OUTPUT_token_with_unk: token_with_unk}, "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_token_unk_agent.print_default_setup_scripts()

