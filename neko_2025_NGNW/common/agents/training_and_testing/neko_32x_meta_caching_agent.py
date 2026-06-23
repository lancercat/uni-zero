import os.path
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent_nograd as awn
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent


import torch

from osocrNG.sptokens import tUNKREP


# problem is supporting caching multiple modality of side info in one caching agent is kinda troublesome,
# so don't do that till we figure out, for now use multiple caching agent if you must
class neko_320_im_meta_caching_agent(neko_module_wrapping_agent):
    PARAM_capacity="capacity";
    PARAM_sinfo_enc_agent="sinfo_enc_agent";  # delegate sinfo_enc_agent_construction to external.
    PARAM_meta_loader_agent="meta_meta_loader_agent"; # meta_meta_loader_agent is now also delegated to external.
    PARAM_sinfo_loader_agent="meta_sinfo_loader_agent"; # meta_sinfo_loader_agent is now also delegated to external.
    PARAM_postprocess_agent="meta_post_process_agent"; # if you need to normalise, arm sp or smth.


    INTR_all_meta_utf="inter_all_meta_utf";
    INTR_all_tdict="inter_all_tdict";
    INTR_batch_utf="inter_batch_utf";
    INTR_batch_tdict="inter_batch_tdict";
    INTR_batch_enc_rawvec="inter_batch_enc_rawvec";
    INTR_batch_enc_utf="inter_batch_enc_utf";
    INTR_all_enc_utf="inter_all_enc_rawvec";
    INTR_all_enc_rawvec="inter_all_enc_utf";


    OUTPUT_tdict_info_path = "tdict_info_path";
    OUTPUT_tdict_info_hash = "tdict_info_hash";
    OUTPUT_armed_proto = "proto";
    OUTPUT_armed_tdict = "tdict";
    OUTPUT_armed_plabel = "plabel";
    OUTPUT_armed_utf = "utf";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.inter_all_meta_utf = neko_get_arg(this.INTR_all_meta_utf,iocvt_dict);
        this.inter_all_tdict = neko_get_arg(this.INTR_all_tdict,iocvt_dict);

        this.inter_batch_utf = neko_get_arg(this.INTR_batch_utf,iocvt_dict);
        this.inter_batch_tdict = neko_get_arg(this.INTR_batch_tdict,iocvt_dict);
        this.inter_batch_enc_rawvec = neko_get_arg(this.INTR_batch_enc_rawvec,iocvt_dict);
        this.inter_batch_enc_utf=neko_get_arg(this.INTR_batch_enc_utf,iocvt_dict);

        this.inter_all_enc_utf = neko_get_arg(this.INTR_all_enc_utf,iocvt_dict);
        this.inter_all_enc_rawvec = neko_get_arg(this.INTR_all_enc_rawvec,iocvt_dict);

        this.plabel = this.register_output(this.OUTPUT_armed_plabel, iocvt_dict);
        this.proto = this.register_output(this.OUTPUT_armed_proto, iocvt_dict);
        this.tdict = this.register_output(this.OUTPUT_armed_tdict, iocvt_dict);
        this.tdict_info_hash = this.register_output(this.OUTPUT_tdict_info_hash, iocvt_dict);
        this.tdict_info_path = this.register_output(this.OUTPUT_tdict_info_path, iocvt_dict);
        this.utf = this.register_output(this.OUTPUT_armed_utf, iocvt_dict);

        pass;


    def set_etc(this, params):
        this.capacity=neko_get_arg(this.PARAM_capacity,params,512);
        this.meta_ldr=this.make(neko_get_arg(this.PARAM_meta_loader_agent,params));
        this.sinfo_ldr=this.make(neko_get_arg(this.PARAM_sinfo_loader_agent,params));
        this.meta_enc=this.make(neko_get_arg(this.PARAM_sinfo_enc_agent,params));
        this.meta_post_proc=this.make(neko_get_arg(this.PARAM_postprocess_agent,params));

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        cws=neko_workspace.empty_like(workspace);
        this.meta_ldr.take_action(cws,environment);
        meta_utfs=cws.get(this.inter_all_meta_utf); # this can be improved.
        meta_tdict=cws.get(this.inter_all_tdict);
        aus=[];
        aps=[];
        # we may not want to keep all templates.
        for i in range(0,len(meta_utfs),this.capacity):
            # that is actually a bigger problem--- gonna need to join the protos and utfs and finally re build tdict.
            butf=meta_utfs[i:i+this.capacity];
            ews=neko_workspace({this.inter_all_meta_utf:butf,this.inter_all_tdict:meta_tdict},devices=workspace.devices);
            this.sinfo_ldr.take_action(ews,environment);
            this.meta_enc.take_action(ews,environment);
            aus+=ews.get(this.inter_batch_enc_utf);
            aps.append(ews.get(this.inter_batch_enc_rawvec));
            # from nep_util.tiling_tensor_vis import process_and_tile_tensors
            # cv2.namedWindow("meow", 0);
            # lten=ews.get(VN.SIDE_INFO_normed_tensor_list(this.PRFX_SINFO));
            # names=ews.get(VN.UTF(this.PRFX_SINFO));
            # im=process_and_tile_tensors(lten,names,(32,32));
            # cv2.imshow("meow",im*254);
            # cv2.waitKey(0);
            pass;
        # during testing we don't keep patches... if you have weird visulization load the pt there---
        # this is to assure that the training and testing code is as scalable as possible....
        cws.add(this.inter_batch_enc_utf,aus);
        cws.add(this.inter_batch_enc_rawvec, torch.cat(aps));
        cid=0;
        sinf_tdict={};
        for i in range(len(aus)): # rearm tdict --- the sideinfo may decide they don't support certain characters.
            prime=meta_tdict[meta_tdict[aus[i]]];
            if(prime not in sinf_tdict):
                sinf_tdict[prime]=cid;
                sinf_tdict[cid] = prime;
                cid+=1;
            sinf_tdict[aus[i]]=sinf_tdict[prime];

        cws.add(this.inter_batch_tdict,sinf_tdict);
        this.meta_post_proc.take_action(cws,environment);
        # force override protocache (bcs its testing.)
        workspace.add(this.tdict,cws.get(this.tdict),True);
        workspace.add(this.proto,cws.get(this.proto),True);
        l=cws.get(this.utf);
        workspace.add(this.utf,l,True); # utf syncs with proto, not logit.
        workspace.add(this.plabel,cws.get(this.plabel),True);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   armed_plabel, armed_proto, armed_tdict, armed_utf, tdict_info_hash, tdict_info_path,
                   capacity, meta_loader_agent,sinfo_loader_agent, sinfo_enc_agent, postprocess_agent,
                   # Added intermediate I/O for consistency
                   inter_all_meta_utf, inter_all_tdict,
                   inter_batch_utf, inter_batch_tdict, inter_batch_enc_rawvec, inter_batch_enc_utf,
                   inter_all_enc_utf, inter_all_enc_rawvec
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {
                # Outputs
                cls.OUTPUT_armed_plabel: armed_plabel, cls.OUTPUT_armed_proto: armed_proto,
                cls.OUTPUT_armed_tdict: armed_tdict, cls.OUTPUT_armed_utf: armed_utf,
                cls.OUTPUT_tdict_info_hash: tdict_info_hash, cls.OUTPUT_tdict_info_path: tdict_info_path,
                # Intermediate I/O from set_mod_io (Needed to satisfy neko_get_arg)
                cls.INTR_all_meta_utf: inter_all_meta_utf,
                cls.INTR_all_tdict: inter_all_tdict,
                cls.INTR_batch_utf: inter_batch_utf,
                cls.INTR_batch_tdict: inter_batch_tdict,
                cls.INTR_batch_enc_rawvec: inter_batch_enc_rawvec,
                cls.INTR_batch_enc_utf: inter_batch_enc_utf,
                cls.INTR_all_enc_utf: inter_all_enc_utf,
                cls.INTR_all_enc_rawvec: inter_all_enc_rawvec
            },
            cls.PARAM_capacity: capacity, cls.PARAM_meta_loader_agent: meta_loader_agent,cls.PARAM_sinfo_loader_agent:sinfo_loader_agent, cls.PARAM_sinfo_enc_agent: sinfo_enc_agent,
            cls.PARAM_postprocess_agent: postprocess_agent, "modcvt_dict": {}
        }};

