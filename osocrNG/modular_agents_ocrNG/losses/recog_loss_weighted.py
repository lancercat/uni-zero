import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

from osocrNG.modular_agents_ocrNG.losses.recog_loss import ocr_loss_agent_mk2



class ocr_loss_agent_mk2_logweight(ocr_loss_agent_mk2):
    INPUT_inst_mapping_name = "inst_mapping_name";
    INPUT_item_log_weight="input_log_weight";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        super().set_mod_io(iocvt_dict, modcvt_dict);
        this.cls_mapping = this.register_input(this.INPUT_inst_mapping_name, iocvt_dict);
        this.item_log_weight=this.register_input(this.INPUT_item_log_weight,iocvt_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel),
            workspace.get(this.cls_mapping)
        );
        weights=torch.exp(workspace.get(this.item_log_weight));
        loss=torch.mean(weights*per_inst_ocrloss);
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname] = terms;
        return workspace, environment;




def get_ocr_loss_agent_mk2_logweight(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,item_log_weight_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name
                                      ):
    engine = ocr_loss_agent_mk2_logweight;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.INPUT_item_log_weight: item_log_weight_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}
class ocr_loss_agent_mk2_logweight_detach_weight(ocr_loss_agent_mk2_logweight):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel),
            workspace.get(this.cls_mapping)
        );
        weights=torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())*0.9+0.1;
        loss=torch.mean(weights*per_inst_ocrloss);
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname] = loss.item();
        return workspace, environment;

def get_ocr_loss_agent_mk2_logweight(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,item_log_weight_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name
                                      ):
    engine = ocr_loss_agent_mk2_logweight;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.INPUT_item_log_weight: item_log_weight_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}
class ocr_loss_agent_mk2_logweight_detach_weight_alter(ocr_loss_agent_mk2_logweight):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel),
            workspace.get(this.cls_mapping)
        );
        with torch.no_grad():
            weights=torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())*0.9+0.1;
        loss=torch.sum(weights*per_inst_ocrloss)/torch.sum(weights);
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname] = loss.item();
        return workspace, environment;


class ocr_loss_agent_mk2_logweight_detach_weight_alter_delayed(ocr_loss_agent_mk2_logweight):
    PARAM_disable_till_eid="disable_till_eid";
    PARAM_disable_till_bid="disable_till_bid";
    def set_etc(this,param):
        super().set_etc(param);
        this.disable_till_eid=neko_get_arg(this.PARAM_disable_till_eid,param);
        this.disable_till_bid = neko_get_arg(this.PARAM_disable_till_bid, param);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        per_inst_ocrloss, terms = environment.module_dict[this.clsloss](
            workspace.get(this.predcls), workspace.get(this.predlen),
            workspace.get(this.clslabel), workspace.get(this.lenlabel),
            workspace.get(this.cls_mapping)
        );
        with torch.no_grad():
            weights=torch.exp(torch.stack(workspace.get(this.item_log_weight)).detach())*0.9+0.1;
        if (workspace.epoch_idx < this.disable_till_eid):
            weights=torch.ones_like(weights);
        elif (workspace.epoch_idx == this.disable_till_eid and workspace.batch_idx < this.disable_till_bid):
            weights=torch.ones_like(weights);
        loss=torch.sum(weights*per_inst_ocrloss)/torch.sum(weights);
        workspace.objdict[this.lossname] = loss;
        workspace.logdict[this.lossname] = loss.item();
        return workspace, environment;

def get_ocr_loss_agent_mk2_logweight_detach_weight(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,item_log_weight_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name
                                      ):
    engine = ocr_loss_agent_mk2_logweight_detach_weight;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.INPUT_item_log_weight: item_log_weight_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}
def get_ocr_loss_agent_mk2_logweight_detach_weight_alter(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,item_log_weight_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name
                                      ):
    engine = ocr_loss_agent_mk2_logweight_detach_weight_alter;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.INPUT_item_log_weight: item_log_weight_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name}}}
def get_ocr_loss_agent_mk2_logweight_detach_weight_alter_delayed(cls_label_name, cls_logit_name, cls_mapping_name,
                                      len_label_name, len_logit_name,item_log_weight_name,
                                      ocr_loss_name,
                                      osocr_loss_mod_name,delay_eid,delay_bid
                                      ):
    engine = ocr_loss_agent_mk2_logweight_detach_weight_alter_delayed;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_cls_label_name: cls_label_name,
                                                      engine.INPUT_cls_logit_name: cls_logit_name,
                                                      engine.INPUT_inst_mapping_name: cls_mapping_name,
                                                      engine.INPUT_len_label_name: len_label_name,
                                                      engine.INPUT_len_logit_name: len_logit_name,
                                                      engine.INPUT_item_log_weight: item_log_weight_name,
                                                      engine.OUTPUT_ocr_loss_name: ocr_loss_name},
                                       "modcvt_dict": {engine.MOD_osocr_loss_mod_name: osocr_loss_mod_name},
                                        engine.PARAM_disable_till_eid:delay_eid,
                                        engine.PARAM_disable_till_bid:delay_bid
                                        }}
