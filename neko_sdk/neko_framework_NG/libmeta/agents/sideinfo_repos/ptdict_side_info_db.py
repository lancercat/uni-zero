from neko_sdk.neko_framework_NG.data.supportdb.support_utf_im_lmdb import neko_random_support_fetcher,neko_first_k_support_fetcher

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.agents.sideinfo_repos.abstract_sinfo_holder_agent import abstract_sideinfo_index_agent
import torch

# use a static representation db. be it token list, text, or just id
# each utf has only one record, if you want be fancy impl you own.
class pt_dict_single_sinfo_agent(abstract_sideinfo_index_agent):
    def set_etc(this, params):
        this.mod_path = neko_get_arg(this.PARAM_mod_path, params);
        this.sinfodict=torch.load(this.mod_path,weights_only=False);

    def refresh_tdict(this,utf,tdict):
        rtdict={};
        id=0;
        for u in utf:
            if u in rtdict:
                continue;
            prime=tdict[tdict[u]];
            if(prime in rtdict):
                rtdict[u]=rtdict[prime];
            else:
                rtdict[id]=prime;
                rtdict[u]=id;
                rtdict[prime]=id;
                id+=1;
        return rtdict;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        utf = workspace.get(this.utf);
        outf=[];
        osinfo=[];
        need_refresh=False;
        for u in utf:
            if(u in this.sinfodict):
                outf.append(u);
                osinfo.append(this.sinfodict[u]);
            else:
                need_refresh=True;
        if(need_refresh):
            tdict=workspace.get(this.tdict)
            rt=this.refresh_tdict(outf,tdict)
            workspace.add(this.sideinfo_tdict,rt);
        else:
            workspace.alias(this.tdict,
                            this.sideinfo_tdict);  # well we are not doing voodoos here so let's not hack our tdict.

        workspace.add(this.sideinfo,osinfo);
        workspace.add(this.sideinfo_utf,outf);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   utf, tdict,
                   sideinfo, sideinfo_utf, sideinfo_tdict,
                   mod_path
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_utf: utf,cls.INPUT_tdict:tdict, cls.OUTPUT_sideinfo: sideinfo, cls.OUTPUT_sideinfo_utf: sideinfo_utf, cls.OUTPUT_sideinfo_tdict: sideinfo_tdict},
            cls.PARAM_mod_path: mod_path,  "modcvt_dict": {}}}


if __name__ == '__main__':
    a=utf_single_lmdb_sideinfo_k_replacing.make(
        utf_single_lmdb_sideinfo_k_replacing.get_agtcfg(
            "utf", "tdict",
            "side-info","sideinfo-utf","sideinfo-tdict",
            "/home/lasercat/ssddata/ctw_ch_yolox/train/", 3
    ));
    ws=neko_workspace({
        "utf" : ["内","卷"],
        "tdict" : {
            "卷":1,
            1:"卷",
            "内": 0,
            0: "内",
    }});
    e=neko_environment();
    a.take_action(ws,e);
    pass;
