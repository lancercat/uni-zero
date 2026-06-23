
from osocrNG.modular_agents_ocrNG.losses.cls_loss import get_per_inst_ocr_cls_loss_agent_mk2
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import neko_symbol_link_alot_agent, \
    neko_symbol_link_alot_agent_weak,neko_symbol_link_agent

from osocrNG.trainable_lossNG.os_clsloss import osclsNG_perinstance,osclsNG_perinstance_d,osclsNG_perinstance_top20_mk2

 # get_translate_agent
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.OSR.classification.agents.neko_missing_token_to_unk import neko_missing_to_unk


from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa

class single_char_cls_mod_factory(virtual_mod_factory):
    def config_loss(this,modcfgdict,prfx):
        modcfgdict = this.config_non_saveable(modcfgdict,osclsNG_perinstance_d,{},
            MN.PER_INSTANCE_OCR_CLS_LOSS_NAME(prfx));

        return modcfgdict;
    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
        localmodprfx=mod_prfx_dict[this.MOD_PRFX_local];
        modcfg=this.config_loss(modcfg,localmodprfx); # always config the loss module--- bcs they don't bite.
        # meta classifier ist not upto the heads' duty to construct and manage--- the meta manager do that
        return modcfg,bogocfg

# gt making is up to loss (? or not?)
class single_char_cls_agt_factory(virtual_agt_factory):
    def get_head_perinst_loss(this, head_training_prfx, loss_mod_prfx):
        return {
            "agent": awa,
            "params": {
                "agent_list": [ "clsloss","lnk"],
                "clsloss" : get_per_inst_ocr_cls_loss_agent_mk2(
                                                                      VN.FLATTEN_ALIGNED_TLABEL(head_training_prfx),
                                                                      VN.FLATTEN_ROI_LOGIT_SEQ(head_training_prfx),
                                                                      VN.FLATTEN_MAP(head_training_prfx),
                                                                      VN.CLS_LOSS_PER_INST(head_training_prfx),
                                                                      MN.PER_INSTANCE_OCR_CLS_LOSS_NAME(loss_mod_prfx)),
                "lnk": neko_symbol_link_agent.get_agtcfg(
                     VN.CLS_LOSS_PER_INST(head_training_prfx),
                    VN.LOSS_PER_INST(head_training_prfx))
                }
        }
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict, agtcfg, agt_prfx, params=None):
        return awa.append_agent_to_cfg(agtcfg,AN.PRED_LO_TR(agt_prfx),this.get_head_perinst_loss(
            data_prfx_dict[this.DATA_PRFX_local],
            mod_prfx_dict[this.MOD_PRFX_local]
        ));

