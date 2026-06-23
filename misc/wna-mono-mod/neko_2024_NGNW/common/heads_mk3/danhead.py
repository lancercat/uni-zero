from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.configs.typical_agent_setups.os_pred import get_pred_agent, get_translate_agent
from osocrNG.configs.typical_agent_setups.aggr import get_temporal_aggr_mk2l,get_seq_flatten
from osocrNG.modular_agents_ocrNG.att_agents.modular_attn.neko_dea_agent import get_neko_basic_temporal_attention_och_global_observation
from osocrNG.modular_agents_ocrNG.losses.cls_loss import get_per_inst_ocr_cls_loss_agent_mk2
from osocrNG.ocr_modules_NG.temporal_att.attention_conv_sigmoid_mk1 import attention_conv_sigmoid_mk1
from neko_sdk.neko_framework_NG.neko_modular_NG import neko_modular_NG as M
from osocrNG.trainable_lossNG.os_clsloss import osclsNG_perinstance,oslenlossNG_perinst
from osocrNG.ocr_modules_NG.lenpred.lenpred_basic_mk1 import lenpred_basic_mk1
from osocrNG.modular_agents_ocrNG.losses.lenpred_loss import get_per_inst_len_loss_agent_mk2
from neko_sdk.neko_framework_NG.agents.utils.ops import neko_sum_agent
from neko_2024_NGNW.common.heads_mk3.head_params import head_common_param
# now it's open to some alternate decoder designs.
from neko_sdk.neko_framework_NG.agents.neko_detacher_agent import get_neko_detacher_agent
from osocrNG.ocr_modules_NG.sampler_NG.dtd_ng_mk1 import neko_DTDNG_mk1, neko_DTDNG_mk1mp
from osocrNG.modular_agents_ocrNG.att_agents.lpred_agent import get_neko_basic_lpred_attention_onemod

class dan_head_factory_mk3:
    MN = mod_names;
    VAN = agent_var_names;
    HP=head_common_param;
    temporal_attention_engine=attention_conv_sigmoid_mk1;
    len_pred_engine=lenpred_basic_mk1;
    def get_lenpred(this,prefix,head_name):
        return get_neko_basic_lpred_attention_onemod(
                    prefix + this.VAN.WORD_TEMP_ENDPOINTS_GPOOL,
                    prefix +head_name+ this.VAN.ATT_LEN_PRED,
                    prefix +head_name+ this.VAN.ATT_LEN_PRED_AMAX,
                    prefix +head_name+ this.MN.WORD_LEN_PRED);

    def get_head_perinst_loss(this,prefix,head_name):
        return {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": ["lenpredloss", "clsloss", "add"],
                "lenpredloss": get_per_inst_len_loss_agent_mk2(
                    prefix + this.VAN.TEN_GT_LEN,
                    prefix + head_name + this.VAN.ATT_LEN_PRED,
                    prefix +head_name+ this.VAN.LEN_LOSS_PER_INSTANCE,
                    prefix +head_name+ this.MN.PER_INSTANCE_OCR_LPRED_LOSS_NAME
                ),
                "clsloss" : get_per_inst_ocr_cls_loss_agent_mk2(
                                                                      prefix + this.VAN.TENSOR_LABEL_NAME,
                                                                      prefix + head_name + this.VAN.LOGITS,
                                                                      prefix + head_name + this.VAN.FLATTEN_MAP,
                                                                      prefix + head_name + this.VAN.CLS_LOSS_PER_INSTANCE,
                                                                      prefix +head_name+ this.MN.PER_INSTANCE_OCR_CLS_LOSS_NAME),
                "add": neko_sum_agent.get_agtcfg(
                    [prefix +head_name+ this.VAN.LEN_LOSS_PER_INSTANCE, prefix +head_name+ this.VAN.CLS_LOSS_PER_INSTANCE],
                    prefix+head_name+this.VAN.LOSS_PER_INSTANCE)
                }
        }

    def get_regularization_term(this,prefix,head_name):
        return None;
    def get_head_perinst_penalty(this,prefix,head_name):
        return get_neko_detacher_agent(prefix+head_name+this.VAN.LOSS_PER_INSTANCE,prefix+head_name+this.VAN.PENALTY_PER_INSTANCE);

    def get_translation(this, prefix, decode_length_tensor_name, gprfx=""):
        return get_translate_agent(prefix + this.VAN.LOGITS, gprfx + this.VAN.TDICT,
                            decode_length_tensor_name, prefix + this.VAN.PRED_TEXT);

    # note this work does not utilize the "global_observation part". This front is a rabbit hole away from publication.
    def get_att(this, prefix,head_name):
        return get_neko_basic_temporal_attention_och_global_observation(prefix + this.VAN.WORD_TEMP_FEAT,prefix+this.VAN.ROUTER_FEAT_NAME,
                                                         prefix +head_name + this.VAN.ATT_MAP, prefix + head_name + this.MN.WORD_ATT);
    def get_aggr(this, prefix,head_name,decode_len_name,sample_from):
        return get_temporal_aggr_mk2l(
                    sample_from,decode_len_name, prefix+head_name + this.VAN.ATT_MAP,
                    prefix+head_name + this.VAN.FULL_WORD_FEAT_SEQ,
                    prefix+head_name + this.MN.DTD_name);  # we may want per-agent drop out policy.
    def get_head(this,prefix,head_name,decode_length_tensor_name,gprfx=""):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [ "len_pred","att","aggr", "flatten","prediction"],
                "len_pred": this.get_lenpred(prefix,head_name),
                "att": this.get_att(prefix,head_name),
                "aggr": this.get_aggr(prefix,head_name,decode_length_tensor_name,prefix+this.VAN.WORD_FEAT),
                "flatten": get_seq_flatten(
                    prefix +head_name+ this.VAN.FULL_WORD_FEAT_SEQ, decode_length_tensor_name,
                    prefix +head_name+ this.VAN.FLATTEN_WORD_FEAT_SEQ, prefix +head_name+ this.VAN.FLATTEN_MAP),
                "prediction":get_pred_agent(prefix +head_name +this.VAN.FLATTEN_WORD_FEAT_SEQ,
                                            prefix+this.VAN.PROTO_VEC , prefix+this.VAN.PROTO_LABEL,
                                            prefix+head_name+this.VAN.LOGITS,gprfx+this.MN.WORD_CLASSIFIER),
            }
        };
        return ac;

    # # don't make the featuremap and class centers sample friendly
    # # usually if you are training some slow-converging unit,
    # # or just want to learn a dedicated locator
    # # you don't want the model to learn wrong correlation due to bad localization
    # def get_isolated_head(this,prefix,head_name,decode_length_tensor_name,gprfx=""):
    #     ac = {
    #         "agent": neko_agent_wrapping_agent,
    #         "params": {
    #             "agent_list": [ "att_aggr", "flatten","prediction"],
    #             "att_aggr": this.get_att_aggr(prefix,head_name,decode_length_tensor_name,prefix+this.VAN.WORD_FEAT_DETACHED),
    #             "flatten": get_seq_flatten(
    #                 prefix +head_name+ this.VAN.FULL_WORD_FEAT_SEQ, decode_length_tensor_name,
    #                 prefix +head_name+ this.VAN.FLATTEN_WORD_FEAT_SEQ, prefix +head_name+ this.VAN.FLATTEN_MAP),
    #             "prediction":get_pred_agent(prefix +head_name +this.VAN.FLATTEN_WORD_FEAT_SEQ,
    #                                         gprfx+this.VAN.PROTO_VEC_DETACHED , gprfx+this.VAN.PROTO_LABEL,
    #                                         prefix+head_name+this.VAN.LOGITS,gprfx+this.MN.WORD_CLASSIFIER),
    #         }
    #     };
    #     return ac;
    def config_local_len_pred(this, modcfgdict, name,params,optparam):
        e = this.len_pred_engine;
        modcfgdict = M.add_config_to_dict(
            modcfgdict,name+this.MN.WORD_LEN_PRED, e, {
                e.PARAM_input_channels: params[this.HP.PARAM_localassch],
                e.PARAM_maxT: params[this.HP.PARAM_maxT],
            }, optparam
        );
        return modcfgdict;
    def config_attn(this, modcfgdict, prefix,params,optparam):
        e=this.temporal_attention_engine;
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+this.MN.WORD_ATT, e,
        {
            e.PARAM_n_parts: params[this.HP.PARAM_nparts],
            e.PARAM_maxT: params[this.HP.PARAM_maxT],
            e.PARAM_number_channels: params[this.HP.PARAM_cam_channels]},optparam);
        return modcfgdict;
    def config_dtd(this, modcfgdict,prefix,params,optparam):
        if (params[this.HP.PARAM_nparts] > 1):
            modtype = neko_DTDNG_mk1mp;
        else:
            modtype = neko_DTDNG_mk1;
        return M.add_config_to_dict(modcfgdict,prefix+this.MN.DTD_name,
            modtype,{},
            optparam);
    def config_loss(this,modcfgdict,prefix,params,optparam):
        modcfgdict = M.add_config_to_dict(modcfgdict, prefix + this.MN.PER_INSTANCE_OCR_CLS_LOSS_NAME,
                                          osclsNG_perinstance, {},None);
        modcfgdict=M.add_config_to_dict(modcfgdict,prefix+ this.MN.PER_INSTANCE_OCR_LPRED_LOSS_NAME,
                                        oslenlossNG_perinst,{},None);
        return modcfgdict;
    def get_head_training_agent_by_string(this,prefix,name,params):
        ac=this.get_head(prefix,name,prefix+ this.VAN.TEN_GT_LEN);
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,name+"loss",
            this.get_head_perinst_loss(prefix,name));
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,name+"regularization",this.get_regularization_term(prefix,name));
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,name+"penalty",
                                                         this.get_head_perinst_penalty(prefix,name));
        return ac;
    def get_head_testing_agent_by_string(this,prefix,name,params):
        ac = this.get_head(prefix, name,prefix+name+this.VAN.ATT_LEN_PRED_AMAX);
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,name + "translate",this.get_translation(
            prefix+name,prefix+name+this.VAN.ATT_LEN_PRED_AMAX,"")
        );
        return ac;
    def config_head(this,modcfgdict,prefix,name,params,opt_params,pprfx=""):
        modcfgdict=this.config_local_len_pred(modcfgdict,prefix+name,params,opt_params);
        modcfgdict= this.config_attn(modcfgdict,prefix+name,params,opt_params);
        modcfgdict = this.config_dtd(modcfgdict, prefix + name, params, opt_params);
        return modcfgdict
    def get_head_mod_by_string(this, modcfg,prefix,name,params,opt_params,pprfx=""):
        return this.config_head(modcfg,prefix,name,params,opt_params,pprfx);


    def get_head_training_extra_mod_by_string(this,modcfg,prefix,name,params,opt_params):
        return this.config_loss(modcfg,prefix+name,params,opt_params);

