from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN,P
from osocrNG.data_utils.data_agents.tokenizer_agents.regex_tokenizer_agent import neko_regex_based_tokenizer_agent
from neko_sdk.cfgtool.argsparse import neko_get_arg
# activelearning gonna wait after we have blackwell together with better io. Prefetching is still the life line here. So we cannot afford gradient
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awa
from osocrNG.lsctNG.agents.configs.basic_lsct_shadow import lsct_shadow_data_agent_factory

from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory

# the lsct is now engaged separately--- different types of LSCT will be built from different factories.
class neko_lsct_shadow_factory(virtual_agt_factory):
    PARAM_seed="seed";
    PARAM_orient="orientation";
    PARAM_charspace="charspace";
    DATA_PRFX_ref="data_ref";
    PARAM_fnt_root="fnt_root";
    PARAM_fnt_idx_file="fnt_idx";
    PARAM_im_root="imroot";
    PARAM_size="size";
    PARAM_lsct_type="lsct_type";


    def get_lsct_data(this,mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        orient = neko_get_arg(this.PARAM_orient, params, [0, 1]);
        size = neko_get_arg(this.PARAM_size, params, 32);
        chaspace = neko_get_arg(this.PARAM_charspace, params, [0, 32, 64]);
        fnt_idx_file = neko_get_arg(this.PARAM_fnt_idx_file, params);
        fnt_root = neko_get_arg(this.PARAM_fnt_root, params);
        imroot = neko_get_arg(this.PARAM_im_root, params);
        type = neko_get_arg(this.PARAM_lsct_type,params,"lsct_repeater");


        dataprfx = neko_get_arg(this.DATA_PRFX_ref, data_prfx_dict);
        lsctprfx = neko_get_arg(this.DATA_PRFX_local, data_prfx_dict);
        if(type=="lsct_repeater"):
            return lsct_shadow_data_agent_factory.get_lsct_repeater(
                VN.UTF(dataprfx), VN.DBG_SELECTED_FNT(lsctprfx), VN.DBG_TFD(lsctprfx), VN.UTF(lsctprfx),
                VN.SAM_FNT(lsctprfx),
                VN.SAM_UPFRONT_MSK_BIN(lsctprfx), VN.SAM_UPFRONT_MSK_INS(lsctprfx), VN.SAM_ORIENT(lsctprfx),
                VN.GT_BIN_MSK(lsctprfx),
                VN.GT_INSSEG_MSK(lsctprfx),VN.GT_INSSEG_UTF(lsctprfx),
                VN.IM_raw(lsctprfx),
                fnt_idx_file, fnt_root, imroot, orient, size, chaspace);
        elif(type=="lsct_shuff"):
            return lsct_shadow_data_agent_factory.get_lsct_shuf(
                VN.UTF(dataprfx), VN.DBG_SELECTED_FNT(lsctprfx), VN.DBG_TFD(lsctprfx), VN.UTF(lsctprfx),
                VN.SAM_FNT(lsctprfx),
                VN.SAM_UPFRONT_MSK_BIN(lsctprfx), VN.SAM_UPFRONT_MSK_INS(lsctprfx), VN.SAM_ORIENT(lsctprfx),
                VN.GT_BIN_MSK(lsctprfx),
                VN.GT_INSSEG_MSK(lsctprfx),VN.GT_INSSEG_UTF(lsctprfx), VN.IM_raw(lsctprfx),
                fnt_idx_file, fnt_root, imroot, orient, size, chaspace);

    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        lsctprfx=neko_get_arg(this.DATA_PRFX_local,data_prfx_dict);

        agtcfg=awa.append_agent_to_cfg(agtcfg,P(lsctprfx,"lsct_maker"),this.get_lsct_data(mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params));

        agtcfg=awa.append_agent_to_cfg(agtcfg,P(lsctprfx,P(lsctprfx,"tokenize")),neko_regex_based_tokenizer_agent.get_agtcfg(
            VN.UTF(lsctprfx), VN.GT_TOK_UTF(lsctprfx)
        )); # the new framework requires the data loader to tokenize.
        # now all lsct repeaters will make semseg masks--- needed or not.
        # cbms is not targeted for eccv.
        # agtcfg=awa.append_agent_to_cfg(agtcfg, P(lsctprfx,"mksemseg"), neko_ins2sem_agent.get_agtcfg(
        #     VN.GT_INSSEG_MSK(lsctprfx), VN.GT_INSSEG_UTF(lsctprfx),
        #     VN.GT_SEMSEG_CBM_MSK(lsctprfx), VN.GT_SEMSEG_CBM_UTF(lsctprfx),
        # ));
        return agtcfg;

