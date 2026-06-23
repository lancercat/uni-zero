import cv2
import torch
# Provides classes to bootstrap testing workspace
# outputs batch data
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.modules.neko_label_sampler_NG import dump_vis_prototype
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.io.data_tiding import neko_aligned_left_top_padding_beacon_np
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder
from osocrNG.names import default_ocr_param_names as dpn
from neko_sdk.neko_framework_NG.names import default_variable_names as dvn
from osocrNG.names import default_ocr_testing_variable_names as dtevn



class neko_single_image_dispatcher_padder(neko_module_wrapping_agent):
    INPUT_raw_image_name=dvn.raw_image_name;
    OUTPUT_raw_beacon_name=dvn.raw_beacon_name;
    OUTPUT_raw_image_size_name=dvn.raw_image_size_name;
    OUTPUT_raw_image_name=dvn.raw_image_name;
    OUTPUT_raw_bmask_name=dvn.raw_bmask_name
    OUTPUT_selector_name=dtevn.selector_name
    def set_etc(this, param):
        this.batch_size = neko_get_arg(dpn.batch_size, param, 1);
        this.anchor_cfgs=neko_get_arg(dpn.anchors,param)
        this.anchor_names = param[dpn.anchors][dpn.anchor_names];
        this.ratio_anchors = [
            param[dpn.anchors][k][dpn.anchor_ratio] for k  in this.anchor_names];
        this.anchors = neko_get_arg(dpn.anchors, param);

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.raw_image_name = this.register(this.INPUT_raw_image_name, iocvt_dict,this.input_dict);
        this.beacon_name = this.register(this.OUTPUT_raw_beacon_name, iocvt_dict,this.output_dict);
        this.size_name=this.register(this.OUTPUT_raw_image_size_name,iocvt_dict,this.output_dict);
        this.image_name=this.register(this.OUTPUT_raw_image_name,iocvt_dict,this.output_dict);
        this.bmask_name=this.register(this.OUTPUT_raw_bmask_name,iocvt_dict,this.output_dict);
        this.anchor_keyword_name=this.register(this.OUTPUT_selector_name,iocvt_dict,this.output_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        raw_im=workspace.inter_dict[this.raw_image_name][0];
        if(len(raw_im.shape)==2):
            raw_im=cv2.cvtColor(raw_im,cv2.COLOR_GRAY2BGR);
        elif(raw_im.shape[-1]==4):
            raw_im=cv2.cvtColor(raw_im,cv2.COLOR_BGRA2BGR);

        h,w,_=raw_im.shape;
        ar=w/h;
        selid=len(this.ratio_anchors)-1; # should every thing fail, it goes to the last anchor.
        for rid in range(len(this.ratio_anchors)):
            if(ar>this.ratio_anchors[rid]):
                selid=rid;
                break;
        w,h=this.anchor_cfgs[this.anchor_names[selid]]["target_size"];
        bw,bh=this.anchor_cfgs[this.anchor_names[selid]]["beacon_size_wh"];

        img,bmask,beacon,sz=neko_aligned_left_top_padding_beacon_np(raw_im,None,h,w,bh,bw);
        workspace.inter_dict[this.beacon_name]=[beacon];
        workspace.inter_dict[this.image_name] = [img];
        workspace.inter_dict[this.bmask_name]=[bmask];
        workspace.inter_dict[this.anchor_keyword_name]=this.anchor_names[selid];
        workspace.inter_dict[this.size_name]=[sz];
        return workspace,environment;


class single_batch_testing_dataset_agent(neko_module_wrapping_agent):
    holder:neko_lmdb_holder;
    INPUT_index_name = "index_name";
    OUTPUT_raw_image_name = dvn.raw_image_name;
    OUTPUT_text_label_name=dvn.raw_label_name;
    def __len__(this):
        return this.holder.nSamples;
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.index_name=this.register(this.INPUT_index_name,iocvt_dict,this.input_dict);
        this.image_name = this.register(this.OUTPUT_raw_image_name, iocvt_dict,this.output_dict);
        this.text_label_name = this.register(this.OUTPUT_text_label_name, iocvt_dict,this.output_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        ret=this.holder.fetch_core(workspace.inter_dict[this.index_name]);
        workspace.inter_dict[this.image_name]=ret["image"];
        workspace.inter_dict[this.text_label_name]=ret["label"];
        return workspace,environment;

# Grabs cached prototype to workspace

class testing_proto_making_agent(neko_module_wrapping_agent):
    MOD_protomvn_name = "proto_mvn_name";
    MOD_prototyper_name = "prototyper_name";
    INPUT_meta_param = dvn.meta_param_name;
    OUTPUT_tensor_proto_vec_name = dvn.tensor_proto_vec_name;
    OUTPUT_rotated_tensor_proto_vec_name = dvn.rotated_tensor_proto_vec_name;
    OUTPUT_weightroto_label_name = dvn.proto_label_name;
    OUTPUT_global_proto_label_name = dvn.global_proto_label_name;

    OUTPUT_tdict_name = dvn.tdict_name;
    OUTPUT_gtdict_name=dvn.gtdict_name;
    def set_etc(this,param):
        this.prototype_cache={};
        this.capacity=neko_get_arg(this.PARAM_capacity,param,256);
        this.possible_rotation=neko_get_arg(this.PARAM_possible_rotation,param,[]);
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.protomvn = this.register(this.MOD_protomvn_name,modcvt_dict,this.mnames);
        this.prototyper = this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.meta_key=this.register(this.INPUT_meta_key_name,iocvt_dict,this.input_dict);

        this.tensor_proto_vec_name=this.register(this.OUTPUT_tensor_proto_vec_name,iocvt_dict,this.output_dict);
        this.rotated_tensor_proto_vec_name=this.register(this.OUTPUT_rotated_tensor_proto_vec_name,iocvt_dict,this.output_dict,"NEP_skipped_NEP");
        this.proto_label_name=this.register(this.OUTPUT_weightroto_label_name,iocvt_dict,this.output_dict)
        this.tdict_name=this.register(this.OUTPUT_tdict_name,iocvt_dict,this.output_dict)
    def make_protos(this,environment,normprotos):
        allprotovec = [];
        for i in range(0, len(normprotos), this.capacity):
            tprotoimg = environment.module_dict[this.protomvn](normprotos[i:i + this.capacity]);
            protos = environment.module_dict[this.prototyper](tprotoimg);
            protos = trnf.normalize(protos, p=2, dim=-1);
            allprotovec.append(protos);
        return torch.cat(allprotovec)
    def take_action(this,workspace,environment:neko_environment):
        with torch.no_grad():
            normprotos, plabels, gplabels, bidict, gbidict=dump_vis_prototype(workspace.inter_dict[this.INPUT_meta_param],False);
            protos=this.make_protos(environment,normprotos);
            workspace.inter_dict[this.OUTPUT_tdict_name]=bidict;
            workspace.inter_dict[this.OUTPUT_gtdict_name]=gbidict;
            workspace.inter_dict[this.OUTPUT_weightroto_label_name]=plabels;
            workspace.inter_dict[this.OUTPUT_global_proto_label_name]=gplabels;

            if (len(this.possible_rotation)):
                this.prototype_cache[m][this.rotated_tensor_proto_vec_name]={0:protos};
                for k in this.possible_rotation:
                    if(k==0):
                        continue;
                    this.prototype_cache[m][this.rotated_tensor_proto_vec_name][k]=\
                        this.make_protos(environment,[torch.rot90(n,k,[0,1]) for n in normprotos]);
        return this.prototype_cache;

class testing_cached_proto_loading_agent(neko_module_wrapping_agent):
    MOD_protomvn_name = "proto_mvn_name";
    MOD_prototyper_name="prototyper_name";
    INPUT_meta_key_name=dvn.meta_key_name;
    OUTPUT_tensor_proto_vec_name=dvn.tensor_proto_vec_name;
    OUTPUT_tensor_proto_img_name=dvn.tensor_proto_img_name;
    OUTPUT_rotated_tensor_proto_vec_name = dvn.rotated_tensor_proto_vec_name;
    OUTPUT_weightroto_label_name = dvn.proto_label_name;
    OUTPUT_tdict_name = dvn.tdict_name;
    PARAM_meta="meta";
    PARAM_capacity="capacity";
    PARAM_possible_rotation="possible_rotation";
    def set_etc(this,param):
        this.metas=param[this.PARAM_meta];
        this.prototype_cache={};
        this.capacity=neko_get_arg(this.PARAM_capacity,param,256);
        this.possible_rotation=neko_get_arg(this.PARAM_possible_rotation,param,[]);
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.protomvn = this.register(this.MOD_protomvn_name,modcvt_dict,this.mnames);
        this.prototyper = this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.meta_key=this.register(this.INPUT_meta_key_name,iocvt_dict,this.input_dict);

        this.tensor_proto_vec_name=this.register(this.OUTPUT_tensor_proto_vec_name,iocvt_dict,this.output_dict);
        this.tensor_proto_img_name=this.register(this.OUTPUT_tensor_proto_img_name,iocvt_dict,this.output_dict);

        this.rotated_tensor_proto_vec_name=this.register(this.OUTPUT_rotated_tensor_proto_vec_name,iocvt_dict,this.output_dict,"NEP_skipped_NEP");
        this.proto_label_name=this.register(this.OUTPUT_weightroto_label_name,iocvt_dict,this.output_dict)
        this.tdict_name=this.register(this.OUTPUT_tdict_name,iocvt_dict,this.output_dict)


    def drop_all_cached(this):
        this.prototype_cache={};
    def make_protos(this,environment,normprotos):
        allprotovec = [];
        allprotoimg=[];
        for i in range(0, len(normprotos), this.capacity):
            tprotoimg = environment.module_dict[this.protomvn](normprotos[i:i + this.capacity]);
            protos = environment.module_dict[this.prototyper](tprotoimg);
            protos = trnf.normalize(protos, p=2, dim=-1);
            allprotovec.append(protos);
            allprotoimg.append(tprotoimg);
        return torch.cat(allprotovec),torch.cat(allprotoimg);
    def cache_prototypes(this,environment:neko_environment):
        with torch.no_grad():
            for m in this.metas:
                normprotos, plabels, gplabels, bidict, gbidict=dump_vis_prototype(this.metas[m],False);
                protos_vec,protos_img=this.make_protos(environment,normprotos);

                this.prototype_cache[m]={
                    this.tensor_proto_img_name:protos_img,
                    this.tensor_proto_vec_name:protos_vec,
                    this.tdict_name:bidict,
                    this.proto_label_name:plabels
                }
                if (len(this.possible_rotation)):
                    this.prototype_cache[m][this.rotated_tensor_proto_vec_name]={0:protos_vec};
                    for k in this.possible_rotation:
                        if(k==0):
                            continue;
                        this.prototype_cache[m][this.rotated_tensor_proto_vec_name][k],_=\
                            this.make_protos(environment,[torch.rot90(n,k,[0,1]) for n in normprotos]);
        return this.prototype_cache;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        proto_key=workspace.inter_dict[this.input_dict.meta_key];
        if(proto_key not in this.prototype_cache):
            pass;
        # Grab translate
        workspace.inter_dict[this.tensor_proto_vec_name]=this.prototype_cache[proto_key]["prototypes"];
        workspace.inter_dict[this.tdict_name] =this.prototype_cache[proto_key]["tdict"];
        workspace.inter_dict[this.proto_label_name] = environment.assets_dict["plabels"];
        return workspace,environment;


class testing_cached_proto_loading_agent_multiple(neko_module_wrapping_agent):
    def set_etc(this,param):
        this.metas=param["meta"];
        this.prototype_cache={};
        this.capacity=neko_get_arg("capacity",param,256);
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.mnames.protomvn = neko_get_arg("proto_mvn_name", modcvt_dict);
        this.mnames.prototypers = neko_get_arg("prototyper_names", modcvt_dict);
        this.input_dict.meta_keys=neko_get_arg("meta_key_names",iocvt_dict);



    def drop_all_cached(this):
        this.prototype_cache={};

    def cache_prototypes(this,environment:neko_environment):
        with torch.no_grad():
            for m in this.metas:
                for p in this.metas["prototypers"]:
                    normprotos, plabels, gplabels, bidict, gbidict=dump_vis_prototype(this.metas[m],False);
                    environment.module_dict[this.mnames.protomvn](normprotos);
                    allprotovec=[];
                    for i in range(0, len(normprotos), this.capacity):
                        tprotoimg = environment.module_dict[this.mnames.protomvn](normprotos[i:i + this.capacity]);
                        protos = environment.module_dict[this.mnames.prototypers[p]](tprotoimg);
                        protos = trnf.normalize(protos, p=2, dim=-1);
                        allprotovec.append(protos);
                    this.prototype_cache[m]={
                        "prototypes": torch.cat(allprotovec),
                        "tdict":bidict,
                        "plabels":plabels
                    }
        return this.prototype_cache;
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        proto_key=workspace.inter_dict[this.input_dict.meta_key];
        if(proto_key not in this.prototype_cache):
            pass;
        # Grab translate
        workspace.inter_dict[this.output_dict.tensor_proto_vec_name]=this.prototype_cache[proto_key]["prototypes"];
        workspace.inter_dict[this.output_dict.tdict_name] =this.prototype_cache[proto_key]["tdict"];
        workspace.inter_dict[this.output_dict.plabels] = environment.assets_dict["plabels"];
        return workspace,environment;
