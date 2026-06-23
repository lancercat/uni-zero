from neko_2024_NGNW.common.agent_pack.fe_mk3.backbone_mk3 import neko_mk3_collate_fe,neko_mk3_collate_fe_static_grid
from neko_2024_NGNW.common.agent_pack.len_pred import neko_gap_fc_counter
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_label_making_agent import get_neko_label_making_agent
from neko_2024_NGNW.common.namescope import mod_names, agent_var_names
from neko_sdk.neko_framework_NG.agents.loss_util.aggr.delayed_log_weighting import \
    get_logweighting_loss_agent_mk2_detach_weight_alter_delayed
from neko_sdk.neko_framework_NG.agents.neko_detacher_agent import get_neko_detacher_agent
from neko_sdk.CnC.aggregate.weighted_tensor_aggr import get_neko_simple_weighted_aggr_agent_nomissing, \
    get_neko_simple_max_conf_aggr_agent_nomissing_list
from osocrNG.modular_agents_ocrNG.att_agents.attn_sel import get_neko_basic_attention_selection_mk2
from neko_2024_NGNW.common.heads_mk3.chaos_head import chaos_head
from neko_2024_NGNW.common.ak6 import AK6
from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import get_neko_symbol_link_agent

from neko_sdk.CnC.common.fork_scores import get_neko_slice_based
from neko_2024_NGNW.common.agent_pack.fe_mk3.tfe_mk3 import neko_mk3_tfe

# mk2 will by default have
# before we unify the behaviour of branch and head, we refrain calling them conquerors
# heads in 310 will be manually given.
# in 320 we seek multihead.
class neko_branch_mk3_single_head:
    MN = mod_names;
    VAN = agent_var_names;
    AGT_W_delay_b = 10000;
    AGT_W_delay_e = 0;

    def append_training_extra(this, ac, prefix):
        # for h in this.head_dict:
        #     ac = neko_agent_wrapping_agent.append_agent_to_cfg(
        #         ac, h+"translation",
        #         this.head_dict[h].get_translation(prefix+h, prefix + this.VAN.ATT_LEN_PRED_AMAX)
        #     );
        return ac;

    def set_bbn_engine(this):
        this.bbn_engine = neko_mk3_collate_fe();

    def set_cntr_engine(this):
        this.cntr_engine = neko_gap_fc_counter();
    def set_tfe_engine(this):
        this.tfe_engine=neko_mk3_tfe();

    def __init__(this):
        this.head_factory = chaos_head();
        this.set_cntr_engine();
        this.set_bbn_engine();
        this.set_tfe_engine();

    def append_mapper(this, ac, prefix, heads):
        if (len(heads[AK6.head_names]) > 1):

            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "predprob",
                                                               get_neko_basic_attention_selection_mk2(
                                                                   prefix + this.VAN.ROUTER_FEAT_NAME,
                                                                   prefix + this.VAN.ATT_SEL_LOGITS,
                                                                   prefix + this.VAN.ATT_SEL_PROB,
                                                                   prefix + this.MN.ATT_SELECTOR
                                                               ));
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "forkprob",
                                                               get_neko_slice_based(prefix + this.VAN.ATT_SEL_PROB,
                                                                                    [prefix + h + this.VAN.ATT_SEL_PROB for
                                                                                     h in heads], -1));
        else:
            pass;
        return ac;

    def append_collector(this, ac, prefix, thing, heads, discrete=False):
        if (len(heads[AK6.head_names]) > 1):
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "_" + thing + "_aggr",
                                                               get_neko_simple_weighted_aggr_agent_nomissing(
                                                                   [prefix + h + this.VAN.ROUTER2_SAM_ID for h
                                                                    in heads[AK6.head_names]],
                                                                   [prefix + h + thing
                                                                    for h in heads[AK6.head_names]],
                                                                   [prefix + h + this.VAN.ATT_SEL_PROB for h in
                                                                    heads[AK6.head_names]],
                                                                   prefix + thing,
                                                                   "NEP_skipped_NEP"
                                                               ));  # expectation of per inst loss
        else:
            # if you don't weight, you don't delay.
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + thing + "_fetch",
                                                               get_neko_symbol_link_agent(
                                                                   prefix +heads[AK6.head_names][0] + thing,
                                                                   prefix + thing
                                                               ));
        return ac;


    def append_list_collector(this, ac, prefix, thing, heads, discrete=False):
        if (len(heads[AK6.head_names]) > 1):
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "_" + thing + "_aggr",
                                                               get_neko_simple_max_conf_aggr_agent_nomissing_list(
                                                                   [prefix + h + this.VAN.ROUTER2_SAM_ID for h
                                                                    in heads[AK6.head_names]],
                                                                   [prefix + h + thing
                                                                    for h in heads[AK6.head_names]],
                                                                   [prefix + h + this.VAN.ATT_SEL_PROB for h in
                                                                    heads[AK6.head_names]],
                                                                   prefix + thing,
                                                                   "NEP_skipped_NEP"
                                                               ));  # expectation of per inst loss
        else:
            # if you don't weight, you don't delay.
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "thing" + "_fetch",
                                                               get_neko_symbol_link_agent(
                                                                   prefix+heads[AK6.head_names][0]+thing,
                                                                   prefix + thing
                                                               ));

        return ac;

    def make_heads_training(this, ac, prefix, heads):
        for head_name in heads[AK6.head_names]:
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(
                ac, head_name, this.head_factory.get_head_training_agent_by_string(
                    prefix,head_name,heads[head_name]["type"],{}));
        return ac;

    def make_heads_testing(this, ac, prefix, heads):
        for head_name in heads[AK6.head_names]:
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(
                ac, head_name, this.head_factory.get_head_testing_agent_by_string(
                    prefix,head_name, heads[head_name]["type"],{}));
        return ac;

    def arm_core(this, ac, prefix,has_tfe):
        fea=this.bbn_engine.get_fp_collate_core(prefix);
        if (has_tfe):
            fea=this.tfe_engine.append_tfe(fea,prefix,"");
        # in mk6 we by default only keep head specific length predictors.
        ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,"core",fea);
        return ac;
    def weight_loss(this, src, dst, weight, base_weight=0.1):
        return get_logweighting_loss_agent_mk2_detach_weight_alter_delayed(weight, src, dst, base_weight,
                                                                           this.AGT_W_delay_e,
                                                                           this.AGT_W_delay_b);
    def get_testing_fp_branch_agent(this, prefix, name, anchorcfg):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [],
                neko_agent_wrapping_agent.PARAM_ACT_VARS: [prefix +name + this.VAN.RAW_IMG_NAME]
            }
        }
        ac = this.arm_core(ac, prefix+name,anchorcfg[AK6.has_tfe]);
        ac=this.make_heads_testing(ac,prefix+name,anchorcfg[AK6.heads]);
        ac = this.append_list_collector(ac, prefix+name, this.VAN.PRED_TEXT,anchorcfg[AK6.heads]);
        return ac;
    def get_training_fp_branch_agent(this, prefix,name, anchorcfg):

        ac = neko_agent_wrapping_agent.empty([prefix +name+ this.VAN.RAW_IMG_NAME]);
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "labelgen",
                                                           get_neko_label_making_agent(
                                                               prefix +name+ this.VAN.PROTO_VEC,
                                                               prefix +name+ this.VAN.RAW_LABEL_NAME,
                                                               prefix +name+ this.VAN.TDICT,
                                                               prefix +name+ this.VAN.TEN_GT_LEN,
                                                               prefix +name+ this.VAN.TENSOR_LABEL_NAME)
                                                           );
        heads=anchorcfg[AK6.heads];
        ac = this.arm_core(ac, prefix+name,anchorcfg[AK6.has_tfe]);
        ac = this.make_heads_training(ac, prefix + name, anchorcfg[AK6.heads]);
        ac = this.append_training_extra(ac, prefix+name);
        ac = this.append_collector(ac, prefix+name, this.VAN.LOSS_PER_INSTANCE,anchorcfg[AK6.heads]);
        ac = this.append_collector(ac, prefix+name, this.VAN.PENALTY_PER_INSTANCE,anchorcfg[AK6.heads]);
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
                                                           prefix + "loss_aggr",
                                                           this.weight_loss(
                                                               prefix +name+ this.VAN.LOSS_PER_INSTANCE,
                                                               prefix +name+ this.VAN.TOTAL_LOSS,
                                                               prefix +name+ this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME));
        return ac;




class neko_branch_mk3(neko_branch_mk3_single_head):

    # override this to have more than one heads.
    # we are cleaning this up, bcs I think we still want it to happen....



    def append_collector(this, ac, prefix, thing, heads, discrete=False):
        if (len(heads[AK6.head_names]) > 1):
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "_" + thing + "_aggr",
                                                               get_neko_simple_weighted_aggr_agent_nomissing(
                                                                   [prefix + h + this.VAN.ROUTER2_SAM_ID for h
                                                                    in heads[AK6.head_names]],
                                                                   [prefix + h + thing
                                                                    for h in heads[AK6.head_names]],
                                                                   [prefix + h + this.VAN.ATT_SEL_PROB for h in
                                                                    heads[AK6.head_names]],
                                                                   prefix + thing,
                                                                   "NEP_skipped_NEP"
                                                               ));  # expectation of per inst loss
        else:
            # if you don't weight, you don't delay.
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + thing + "_fetch",
                                                               get_neko_symbol_link_agent(
                                                                   [prefix + h + thing for h in
                                                                    heads[["names"]]][0],
                                                                   prefix + thing
                                                               ));
        return ac;

    def append_list_collector(this, ac, prefix, thing, heads, discrete=False):
        if (len(heads) > 1):
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + thing + "_aggr",
                                                               get_neko_simple_max_conf_aggr_agent_nomissing_list(
                                                                   [prefix + h + this.VAN.ROUTER2_SAM_ID for h
                                                                    in heads],
                                                                   [prefix + h + thing
                                                                    for h in heads],
                                                                   [prefix + h + this.VAN.ATT_SEL_PROB for h in
                                                                    heads],
                                                                   prefix + thing,
                                                                   "NEP_skipped_NEP"
                                                               ));  # expectation of per inst loss
        else:
            # if you don't weight, you don't delay.
            ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "thing" + "_fetch",
                                                               get_neko_symbol_link_agent(
                                                                   [prefix + h + thing for h in
                                                                    heads][0],
                                                                   prefix + thing
                                                               ));

        return ac;

    # really simple exhaustive mapping
    # will switch to full power routing based mapping after AAAI.
    # so it just fork the probability to the task.
    # by default maps detached prob
    def append_mapper(this, ac, prefix, heads):
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "predprob",
                                                           get_neko_basic_attention_selection_mk2(
                                                               prefix + this.VAN.ROUTER_FEAT_NAME,
                                                               prefix + this.VAN.ATT_SEL_LOGITS,
                                                               prefix + this.VAN.ATT_SEL_PROB,
                                                               prefix + this.MN.ATT_SELECTOR
                                                           ));
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "forkprob",
                                                           get_neko_slice_based(prefix + this.VAN.ATT_SEL_PROB,
                                                                                [prefix + h + this.VAN.ATT_SEL_PROB for
                                                                                 h in heads], -1));
        return ac;



    def get_testing_fp_branch_agent(this, prefix, name, anchorcfg):
        ac = {
            "agent": neko_agent_wrapping_agent,
            "params": {
                "agent_list": [],
                neko_agent_wrapping_agent.PARAM_ACT_VARS: [prefix + this.VAN.RAW_IMG_NAME]
            }
        }
        ac = this.arm_core(ac, prefix+name,anchorcfg[AK6.has_tfe] );
        ac=this.make_heads_testing(ac,prefix+name,anchorcfg[AK6.heads]);
        ac = this.append_list_collector(ac, prefix+name,this.VAN.PRED_TEXT,anchorcfg[AK6.heads][AK6.head_names]);
        return ac;

    # if this thing has more than one heads, overwrite this method.

    def append_len_loss(this, ac, prefix):
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "lenloss",
                                                           this.cntr_engine.get_len_loss(prefix)
                                                           );
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
                                                           prefix + "lenaggr",
                                                           this.weight_loss(
                                                               prefix + this.VAN.LEN_LOSS_PER_INSTANCE,
                                                               prefix + this.VAN.LEN_LOSS,
                                                               prefix + this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME));
        return ac

    def append_loss_aggr(this, ac, prefix):
        pass;

    def append_perinst_loss(this, ac, prefix, heads):

        ac = this.append_collector(ac, prefix,heads, this.VAN.CLS_LOSS_PER_INSTANCE);
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
                                                           prefix + "clsaggr",
                                                           this.weight_loss(
                                                               prefix + this.VAN.CLS_LOSS_PER_INSTANCE,
                                                               prefix + this.VAN.CLS_LOSS,
                                                               prefix + this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME));

        return ac;

    # say if you want to align heads, do some contrast learning magic....

    def append_branch_feasibility(this, ac, prefix):
        # detach as a feasibility metric
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
                                                           prefix + this.VAN.ANAME_punisher,
                                                           get_neko_detacher_agent(
                                                               [prefix + this.VAN.CLS_LOSS_PER_INSTANCE],
                                                               [prefix + this.VAN.BRANCH_PENALTY]));

        return ac;

    def append_head_feasibility(this, ac, prefix):
        return ac;

    def append_head_choice_loss(this, ac, prefix):
        return ac;

    def get_training_fp_branch_agent(this, prefix,name,anchorcfg):
        ac = neko_agent_wrapping_agent.empty([prefix +name+ this.VAN.RAW_IMG_NAME]);
        ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "labelgen",
                                                           get_neko_label_making_agent(
                                                               prefix +name + this.VAN.PROTO_VEC,
                                                               prefix  +name+ this.VAN.RAW_LABEL_NAME,
                                                               prefix  +name+ this.VAN.TDICT,
                                                               prefix  +name+ this.VAN.TEN_GT_LEN,
                                                               prefix  +name+ this.VAN.TENSOR_LABEL_NAME)
                                                           );

        ac = this.arm_core(ac, prefix+name,anchorcfg[AK6.has_tfe] );
        # losses and rewards are now armed within heads.
        if (len(anchorcfg[AK6.heads][AK6.head_names]) > 1):
            ac = this.append_head_feasibility(ac, prefix);
            ac = this.append_head_choice_loss(ac,
                                              prefix);  # if you choose head based on expection over loss, ignore these two functions as they do nothing by default.
        ac = this.append_branch_feasibility(ac, prefix);
        ac = this.append_training_extra(ac, prefix);  # if you have something need

        return ac;

class neko_branch_mk3_single_head_grid_coll(neko_branch_mk3_single_head):
    def set_bbn_engine(this):
        this.bbn_engine = neko_mk3_collate_fe_static_grid();


#
# from neko_2024_NGNW.common.agent_pack.backbone import neko_mk5_ii_collate_fe,neko_mk5_ii_trainable_collate_fe
# from neko_2024_NGNW.common.agent_pack.len_pred import neko_gap_fc_counter
# from neko_2024_NGNW.common.heads_mk2.chaos_head import chaos_head
# from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
# from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_label_making_agent import get_neko_label_making_agent
# from neko_2024_NGNW.common.namescope import mod_names,agent_var_names
# from neko_sdk.neko_framework_NG.agents.loss_util.aggr.delayed_log_weighting import get_logweighting_loss_agent_mk2_detach_weight_alter_delayed
# from osocrNG.configs.typical_agent_setups.os_pred import get_pred_agent, get_translate_agent
# from neko_sdk.neko_framework_NG.agents.neko_detacher_agent import get_neko_detacher_agent
# from neko_sdk.CnC.aggregate.weighted_tensor_aggr import get_neko_simple_weighted_aggr_agent_nomissing,get_neko_simple_max_conf_aggr_agent_nomissing_list
# from neko_sdk.neko_framework_NG.agents.utils.symbol_link_agent import get_neko_symbol_link_agent
# from osocrNG.modular_agents_ocrNG.att_agents.attn_sel import get_neko_basic_attention_selection_mk2
# from neko_sdk.CnC.rewards.supervise_on_min_loss_mk2 import neko_supervised_on_exhaustive_min_penalty
# from osocrNG.modular_agents_ocrNG.metric_agents.ned import get_neko_ned_agent_as_penalization
# from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_label_censor_agent import get_neko_label_censor_agent
#
# from neko_sdk.CnC.common.fork_scores import get_neko_slice_based
# # mk2 will by default have
# # before we unify the behaviour of branch and head, we refrain calling them conquerors
# # an mk3 branch takes a batch of images, produce a batch of aggregated results/logits, together with a bunch of head predictions/logits.
# # intermediate features will also be put ot the workspace.
#
# class neko_branch_mk3:
#     MN = mod_names;
#     VAN = agent_var_names;
#     AGT_W_delay_b = 10000;
#     AGT_W_delay_e = 0;
#
#
#     # mk3 branch have master heads, servant heads, and additional heads.
#     # master heads are counted for penalty and subjects to selection
#     # keywords are guarded with nosense -- you don't worry your head name stumble on one of the keywords.
#     TYPE_MASTER="NEP_master_NEP";
#     # servant heads are treated as regularization terms and subjects to selection
#     # they won't affect higher level of selection, but when the current branch is selected,
#     # they can be picked up by attention selector
#     TYPE_SERVANT = "NEP_servant_NEP";
#     # additional heads are just regularization terms that will not be selected
#     # (the test will include additional heads anyway for debugging purpose but during production you need to rid of them)
#     TYPE_ADDITION = "NEP_addition_NEP";
#
#     TYPE_ANY="NEP_all_NEP";
#     TYPE_SELECTABLE="NEP_selectable_NEP";
#
#     PARAM_skip_counter="NEP_skip_counter_NEP";
#     PARAM_skip_tfe="NEP_skip_tfe_NEP";
#     def register_heads(this,hdict,type,names,types):
#         for n,t in zip(names,types):
#             assert (n not in hdict);
#             hdict[type].append(n);
#             hdict[n]=this.head_factory.get_head_agent_by_string(t);
#         return hdict;
#
#     # override this to have more than one heads.
#     # we are cleaning this up, bcs I think we still want it to happen....
#     def get_head_dict(this,masters,master_types,
#                                         servants, servant_types,
#                                         additions,addition_types):
#         heads={
#             this.TYPE_MASTER:[],
#             this.TYPE_SERVANT:[],
#             this.TYPE_ADDITION:[]
#         };
#         heads=this.register_heads(heads,cls.TYPE_MASTER,masters,master_types);
#         heads = this.register_heads(heads, cls.TYPE_SERVANT, servants, servant_types);
#         heads = this.register_heads(heads, cls.TYPE_ADDITION, additions, addition_types);
#         heads[this.TYPE_ANY]=heads[cls.TYPE_MASTER]+heads[cls.TYPE_SERVANT]+heads[cls.TYPE_ADDITION];
#         heads[this.TYPE_SELECTABLE]=heads[cls.TYPE_MASTER]+heads[cls.TYPE_SERVANT]+heads[cls.TYPE_ADDITION]
#         return heads;
#     # there will be no separated "aux_heads". Everything is in headdict now.
#     def make_heads(this,head_dict,ac,prefix,decode_length_tensor_name,gprfx):
#         for head_name in head_dict[this.TYPE_ANY]:
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "head"+head_name,
#                                                        head_dict[head_name].get_head(prefix,head_name, decode_length_tensor_name,gprfx));
#         return ac;
#
#     def set_bbn_engine(this):
#         this.bbn_engine=neko_mk5_ii_collate_fe();
#     def set_cntr_engine(this):
#         this.cntr_engine=neko_gap_fc_counter();
#
#     def get_variants(this):
#         return {"": {"pred": this.VAN.PRED_TEXT}};
#
#     def __init__(this):
#         this.set_bbn_engine();
#         this.set_tfe_engine();
#         this.set_cntr_engine();
#
#         if(this.cntr_engine is not None):
#             assert (this.tfe_engine is not None); # if you do not have temporal engine, you cannot count.
#
#     def weight_loss(this, src, dst, weight,base_weight=0.1):
#         return get_logweighting_loss_agent_mk2_detach_weight_alter_delayed(weight, src, dst, base_weight, this.AGT_W_delay_e,
#                                                                            this.AGT_W_delay_b);
#
#     def get_translation(this, prefix, decode_length_tensor_name, gprfx=""):
#         return get_translate_agent(prefix + this.VAN.LOGITS, gprfx + this.VAN.TDICT,
#                             decode_length_tensor_name, prefix + this.VAN.PRED_TEXT);
#     def append_item_collector(this,ac,prefix,thing,selectable,rangestr="",discrete=False):
#         if (len(selectable) > 1):
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + thing+"_aggr"+rangestr,
#                                                                        get_neko_simple_weighted_aggr_agent_nomissing(
#                                                                            [prefix + h + this.VAN.ROUTER2_SAM_ID for h in selectable],
#                                                                            [prefix + h + thing for h in selectable],
#                                                                            [prefix + h + this.VAN.ATT_SEL_PROB for h in selectable],
#                                                                            prefix + thing,
#                                                                            "NEP_skipped_NEP"
#                                                                        ));  # expectation of per inst loss
#         else:
#             # if you don't weight, you don't delay.
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + thing+"_fetch"+rangestr,
#                                                                get_neko_symbol_link_agent(
#                                                                    [selectable][0],
#                                                                    prefix + thing
#                                                                ));
#
#
#         return ac;
#     def append_loss_collector(this,head_dict,ac,prefix,thing):
#         selectable=head_dict[this.TYPE_ANY];
#         return this.append_item_collector(ac,prefix,thing,selectable,rangestr="_any");
#     def append_local_penalty_collector(this,head_dict,ac,prefix,thing):
#         selectable = head_dict[this.TYPE_SELECTABLE];
#         return this.append_item_collector(ac, prefix, thing, selectable,rangestr="_selectable");
#     def append_global_penalty_collector(this,head_dict,ac,prefix,thing):
#         selectable = head_dict[this.TYPE_MASTER];
#         return this.append_item_collector(ac, prefix, thing, selectable,rangestr="_master");
#
#     def append_list_collector(this,head_dict,ac,prefix,thing,selectable,discrete=False):
#         if (len(head_dict) > 1):
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix +"_"+ thing+"_aggr",
#                                                                        get_neko_simple_max_conf_aggr_agent_nomissing_list(
#                                                                            [prefix + h + this.VAN.ROUTER2_SAM_ID for h
#                                                                             in head_dict],
#                                                                            [prefix + h + thing
#                                                                             for h in head_dict],
#                                                                            [prefix + h + this.VAN.ATT_SEL_PROB for h in
#                                                                             head_dict],
#                                                                            prefix + thing,
#                                                                            "NEP_skipped_NEP"
#                                                                        ));  # expectation of per inst loss
#         else:
#             # if you don't weight, you don't delay.
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "thing"+"_fetch",
#                                                                get_neko_symbol_link_agent(
#                                                                    [prefix + h + thing for h in
#                                                                     head_dict][0],
#                                                                    prefix + thing
#                                                                ));
#
#
#         return ac;
#     # really simple exhaustive mapping
#     # will switch to full power routing based mapping after AAAI.
#     # so it just fork the probability to the task.
#     # by default maps detached prob
#     def append_mapper(this,head_dict,ac,prefix):
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "predprob",
#                                                            get_neko_basic_attention_selection_mk2(
#                                                                prefix + this.VAN.ROUTER_FEAT_NAME,
#                                                                prefix + this.VAN.ATT_SEL_LOGITS,
#                                                                prefix + this.VAN.ATT_SEL_PROB,
#                                                                prefix + this.MN.ATT_SELECTOR
#                                                            ));
#         ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac,"forkprob",
#                                                          get_neko_slice_based(prefix + this.VAN.ATT_SEL_PROB,
#                                             [prefix + h + this.VAN.ATT_SEL_PROB for h in head_dict], -1));
#         return ac;
#     def arm_core(this,head_dict,ac,prefix,decode_length_tensor_name,gprfx=""):
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, "fe", this.bbn_engine.get_fp_collate_core(prefix));
#         if(head_dict[this.PARAM_skip_tfe] or this.tfe_engine is None):
#             pass;
#         else:
#             ac = this.tfe_engine.append_tfe(ac,prefix);
#         if(head_dict[this.PARAM_skip_counter] or this.cntr_engine is None):
#             pass;
#         else:
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,  "length_head", this.cntr_engine.get_lenpred(prefix));
#         if(len(head_dict)>1):
#             ac=this.append_mapper(head_dict,ac,prefix);
#         this.make_heads(ac,head_dict,prefix,decode_length_tensor_name,gprfx);
#         return ac;
#
#     def get_testing_fp_branch_agent(this,head_dict,prefix):
#         ac = {
#             "agent": neko_agent_wrapping_agent,
#             "params": {
#                 "agent_list": [],
#                 neko_agent_wrapping_agent.PARAM_ACT_VARS: [prefix + this.VAN.RAW_IMG_NAME]
#             }
#         }
#         ac = this.arm_core(ac, prefix,prefix+this.VAN.ATT_LEN_PRED_AMAX);
#         for h in head_dict:
#             ac = neko_agent_wrapping_agent.append_agent_to_cfg(
#                 ac, h+"translation",
#                 head_dict[h].get_translation(prefix+h, prefix + this.VAN.ATT_LEN_PRED_AMAX)
#             );
#         ac= this.append_list_collector(ac,prefix,this.VAN.PRED_TEXT,head_dict[this.TYPE_SELECTABLE]);
#         return ac;
#     # if this thing has more than one heads, overwrite this method.
#
#
#     def append_len_loss(this,head_dict,ac,prefix):
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + "lenloss",
#                                                            this.cntr_engine.get_len_loss(prefix)
#                                                            );
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
#                                                            prefix + "lenaggr",
#                                                            this.weight_loss(
#                                                                prefix + this.VAN.LEN_LOSS_PER_INSTANCE,
#                                                                prefix + this.VAN.LEN_LOSS,
#                                                                prefix + this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME));
#         return ac
#
#     def append_loss_aggr(this,head_dict,ac,prefix):
#         pass;
#
#
#     def append_perinst_loss(this,head_dict,ac,prefix):
#         # again if you don't count, you don't learn how.
#         if(this.cntr_engine is not None):
#             ac = this.append_len_loss(ac, prefix);
#         for h in head_dict:
#             ac=neko_agent_wrapping_agent.append_agent_to_cfg(ac, prefix + h + "clsloss",
#                head_dict[h].get_head_perinst_loss(prefix,h));
#         ac=this.append_loss_collector(ac,prefix,this.VAN.CLS_LOSS_PER_INSTANCE);
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
#                                                            prefix+"clsaggr",
#                                                            this.weight_loss(
#                                                                prefix + this.VAN.CLS_LOSS_PER_INSTANCE,
#                                                                prefix  + this.VAN.CLS_LOSS,
#                                                                prefix  + this.VAN.DETACHED_ROUTER_PATH_LOG_PROB_NAME));
#
#
#         return ac;
#     # say if you want to align heads, do some contrast learning magic....
#     def append_training_extra(this,head_dict,ac,prefix):
#         # for h in head_dict:
#         #     ac = neko_agent_wrapping_agent.append_agent_to_cfg(
#         #         ac, h+"translation",
#         #         head_dict[h].get_translation(prefix+h, prefix + this.VAN.ATT_LEN_PRED_AMAX)
#         #     );
#         return ac;
#
#     def append_branch_feasibility(this,head_dict, ac, prefix):
#         ac = this.append_loss_collector(ac, prefix, this.VAN.CLS_LOSS_PER_INSTANCE);
#
#         # detach as a feasibility metric
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,
#                                                            prefix  + this.VAN.ANAME_punisher,
#                                                            get_neko_detacher_agent(
#                                                                [prefix  + this.VAN.CLS_LOSS_PER_INSTANCE],
#                                                                [prefix + this.VAN.BRANCH_PENALTY]));
#
#         return ac;
#     def append_head_feasibility(this,head_dict,ac,prefix):
#         return ac;
#     def append_head_choice_loss(this,head_dict,ac,prefix):
#         return ac;
#
#     def get_training_fp_branch_agent(this, head_dict,prefix):
#         ac = neko_agent_wrapping_agent.empty([prefix  + this.VAN.RAW_IMG_NAME]);
#         ac = neko_agent_wrapping_agent.append_agent_to_cfg(ac,"labelgen",
#             get_neko_label_making_agent(
#             prefix  + this.VAN.PROTO_VEC,
#             prefix + this.VAN.RAW_LABEL_NAME,
#             prefix  + this.VAN.TDICT,
#             prefix  + this.VAN.TEN_GT_LEN,
#             prefix  + this.VAN.TENSOR_LABEL_NAME)
#        );
#
#         ac = this.arm_core(ac, prefix,prefix+this.VAN.TEN_GT_LEN);
#         ac = this.append_perinst_loss(ac,prefix);
#         if(len(head_dict)>1):
#             ac=this.append_head_feasibility(ac,prefix);
#             ac=this.append_head_choice_loss(ac,prefix); # if you choose head based on expection over loss, ignore these two functions as they do nothing by default.
#         ac=this.append_branch_feasibility(ac,prefix);
#         ac=this.append_training_extra(ac,prefix); # if you have something need
#
#         return ac;
#
# # if you have no
# class neko_branch_mk3_ntf:
#     pass;
