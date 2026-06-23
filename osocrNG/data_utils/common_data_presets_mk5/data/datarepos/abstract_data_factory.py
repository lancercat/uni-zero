# although its possible to mux augmentation node for different sources---- let's not do that bcs its unnecessarily complex

from osocrNG.data_utils.data_agents.tokenizer_agents.regex_tokenizer_agent import neko_regex_based_tokenizer_agent
from osocrNG.data_utils.common_data_presets_mk4.typical_setups.varnames import DATA_VN as VN
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from osocrNG.data_utils.common_data_presets_mk4.presets.names import P
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_data_factory_mk5

class abstract_data_repo_mixin(neko_virtual_data_factory_mk5):
    PARAM_split_pattern="split_pattern";
    @classmethod
    def arm_disldr(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        return acfg
    @classmethod
    def arm_tokenizer(cls,acfg,data_prfxs,mod_prfxs,agt_prfxs,params):
        split_pattern=neko_get_arg(cls.PARAM_split_pattern,params,r"\X");
        data_prefix = data_prfxs[cls.DATA_PRFX];
        aprfx = neko_get_arg(cls.AGT_PRFX, agt_prfxs, data_prefix);
        ta=awa.wrap_this(neko_regex_based_tokenizer_agent.get_agtcfg(
            VN.UTF(data_prefix), VN.GT_TOK_UTF(data_prefix),split_pattern),actvars=[VN.UTF(data_prefix)]);
        return awa.append_agent_to_cfg(acfg, P(aprfx,"tokenizer"),ta);

    @classmethod
    def arm_data_repo(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        acfg=cls.arm_disldr(acfg,data_prfxs,mod_prfxs,agt_prfxs,params);
        acfg=cls.arm_tokenizer(acfg,data_prfxs,mod_prfxs,agt_prfxs,params);
        return acfg;


