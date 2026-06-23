import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_simple_xtcs_agent(ama):
    PARAM_TDICT="tdict";
    INPUT_seq_feat_list="feat_seq_list";
    INPUT_seq_utf_list="seq_utf_list";
    INPUT_item_feat_list="item_feat_list";
    INPUT_item_utf_list="item_utf_list";
    INPUT_frozen_item_feat_list="frozen_item_feat_list";
    INPUT_frozen_item_utf_list="frozen_item_utf_list";

    OUTPUT_cos_sim_loss="sim_loss"

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.frozen_item_feat_list = this.register_input(this.INPUT_frozen_item_feat_list, iocvt_dict);
        this.frozen_item_utf_list = this.register_input(this.INPUT_frozen_item_utf_list, iocvt_dict);
        this.item_feat_list = this.register_input(this.INPUT_item_feat_list, iocvt_dict);
        this.item_utf_list = this.register_input(this.INPUT_item_utf_list, iocvt_dict);
        this.seq_feat_list = this.register_input(this.INPUT_seq_feat_list, iocvt_dict);
        this.seq_utf_list = this.register_input(this.INPUT_seq_utf_list, iocvt_dict);
        this.cos_sim_loss = this.register_output(this.OUTPUT_cos_sim_loss, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.tdict = torch.load(neko_get_arg(this.PARAM_TDICT, params),weights_only=False);
        cdict={};
        uks=[];
        for i in this.tdict:
            if(type(i)==str):
                id=this.tdict[i];
                if(id not in cdict):
                    cdict[id]=[];
                cdict[id].append(i);
        for k in cdict:
            if (len(cdict[k])==1):
                uks+=cdict[k];
        this.uks=set(uks);

    def register_seq_to_dict(this,vds,workspace,uname,fname, detach):
        if(not workspace.has(fname)):
            return vds;
        if(not detach):
            fseq=workspace.get(fname);
        else:
            fseq=workspace.get(fname).detach();
        useq=workspace.get(uname);
        for sid in range(len(useq)):
            for uid in range(min(fseq.shape[1],len(useq[sid]))):
                tok=useq[sid][uid];
                if(tok not in this.uks):
                    continue;
                vec=fseq[sid][uid];
                if(tok not in vds):
                    vds[tok]=[];
                vds[tok].append(vec);
        return vds;
    def register_item_to_dict(this,vds,workspace,uname,fname, detach):
        if(not workspace.has(fname)):
            return vds;
        if (not detach):
            fvs = workspace.get(fname);
        else:
            fvs = workspace.get(fname).detach();
        utfs=workspace.get(uname);

        for sid in range(len(utfs)):
            tok=utfs[sid];
            if(tok not in this.uks):
                continue;
            vec=fvs[sid];
            if(tok not in vds):
                vds[tok]=[];
            vds[tok].append(vec);
        return vds;
    def mv_smtx(this,vs):
        nvs = torch.nn.functional.normalize(torch.cat(vs).reshape(len(vs), -1), dim=-1, p=2);
        mv = torch.nn.functional.normalize(nvs.mean(0, keepdim=True), dim=1, p=2);
        smtx = nvs.matmul(nvs.T);
        return mv,smtx;

    def aggr(this,smtx):
        triu=torch.triu(torch.ones_like(smtx),diagonal=1);
        mdist=(triu*smtx).sum()/torch.clip(triu.sum(),0.0000001);
        return mdist;

    def aggr_th(this,smtx,thr=0.5):
        triu=torch.triu(torch.ones_like(smtx),diagonal=1);
        mdist=(triu*torch.nn.functional.relu(smtx-thr)).sum()/torch.clip(triu.sum(),0.0000001);
        return mdist;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        vds={};
        for fn,un in zip(this.frozen_item_feat_list,this.frozen_item_utf_list):
            vds=this.register_item_to_dict(vds,workspace,un,fn,True);
        for fn,un in zip(this.item_feat_list,this.item_utf_list):
            vds=this.register_item_to_dict(vds,workspace,un,fn,False);
        for fn,un in zip(this.seq_feat_list,this.seq_utf_list):
            vds=this.register_seq_to_dict(vds,workspace,un,fn,False);
        ms=[];
        mdists=[];
        for k in vds:
            mv,smtx=this.mv_smtx(vds[k]);
            ms.append(mv);
            mdists.append(this.aggr(smtx));

        intra=torch.stack(mdists).mean();
        _,smtx=this.mv_smtx(ms);
        inter=this.aggr_th(smtx);
        workspace.add_loss(this.cos_sim_loss,inter-intra);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   frozen_item_feat_list, frozen_item_utf_list, item_feat_list, item_utf_list, seq_feat_list,
                   seq_utf_list,
                   cos_sim_loss,
                   TDICT
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.INPUT_frozen_item_feat_list: frozen_item_feat_list,
                                                        cls.INPUT_frozen_item_utf_list: frozen_item_utf_list,
                                                        cls.INPUT_item_feat_list: item_feat_list,
                                                        cls.INPUT_item_utf_list: item_utf_list,
                                                        cls.INPUT_seq_feat_list: seq_feat_list,
                                                        cls.INPUT_seq_utf_list: seq_utf_list,
                                                        cls.OUTPUT_cos_sim_loss: cos_sim_loss}, cls.PARAM_TDICT: TDICT,
                                         "modcvt_dict": {}}}



if __name__ == '__main__':
    from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
    ws=torch.load("/run/media/lasercat/data/cat/neko_wcki/neko_2025_NGNW/beta320/b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-XTCS/a.pt",weights_only=False);
    ac=neko_simple_xtcs_agent.get_agtcfg(
        [],[],['meta-ostrv1-tr-sampled-noto-armed-im_feature_vector'],['meta-ostrv1-tr-sampled-noto-armed-utf'],['ostrv1-abiaug-roi_feat_sequential'],['ostrv1-abiaug-sample_utf_tokenized'],
        "debug_simloss",
        "/home/lasercat/ssddata/dicts_v3/chs3755_en_uncased_digits/tdict.pt"
    );
    a=neko_simple_xtcs_agent.make(ac);
    e=neko_environment();
    a.take_action(ws,e);
    pass;


