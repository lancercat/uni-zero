import cv2

from neko_sdk.neko_framework_NG.data.supportdb.support_utf_im_lmdb import neko_random_support_fetcher,neko_first_k_support_fetcher

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.libmeta.agents.sideinfo_repos.abstract_sinfo_holder_agent import abstract_sideinfo_index_agent


# load from precached images and fetch with utf
# More versions will come if needed
# bcs you may need to join a few more repos instead making a huge repo--- there are infinite combinations--- better to be agile
class utf_single_lmdb_sideinfo_k_replacing(abstract_sideinfo_index_agent):
    PARAM_k="k_for_each_utf";
    PARAM_rand="rand";
    PARAM_seed="seed";
    PARAM_force_grey="force_grey"
    def set_etc(this, params):
        this.mod_path = neko_get_arg(this.PARAM_mod_path, params);
        this.k=neko_get_arg(this.PARAM_k,params,3);
        this.rand=neko_get_arg(this.PARAM_rand,params,True);
        this.force_grey=neko_get_arg(this.PARAM_force_grey,params,False);
        this.seed=None;
        if(this.rand):
            this.seed=neko_get_arg(this.PARAM_seed,params,9)
        names=list(this.mod_path.keys());

        if(this.rand):
            this.holder=neko_random_support_fetcher({
                neko_random_support_fetcher.PARAM_root_names:names,
                neko_random_support_fetcher.PARAM_root_dict:this.mod_path,
                neko_random_support_fetcher.PARAM_seed: this.seed
            });
        else:
            this.holder = neko_first_k_support_fetcher({
                neko_random_support_fetcher.PARAM_root_names: names,
                neko_random_support_fetcher.PARAM_root_dict: this.mod_path,
            });

        pass;
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
                rtdict[id]=u;
                rtdict[u]=id;
                id+=1;
        return rtdict;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        utf = workspace.get(this.utf);
        outf=[];
        osinfo=[];
        need_refresh=False;
        for u in utf:
            suppsam=this.holder.fetch(u,this.k);
            if suppsam is None: # if there is no sideinfo for this et al.
                need_refresh=True;
            else:
                if(this.force_grey):
                    suppsam=[cv2.cvtColor(cv2.cvtColor(s,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR) for s in suppsam]; # force using gray
                osinfo+=suppsam;
                outf+=[u for _ in suppsam]; # if it exists but fails to deliver k--- hu knows?
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
                   mod_path,k=3,rand=True,seed=9,force_grey=False,
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_utf: utf,cls.INPUT_tdict:tdict, cls.OUTPUT_sideinfo: sideinfo, cls.OUTPUT_sideinfo_utf: sideinfo_utf, cls.OUTPUT_sideinfo_tdict: sideinfo_tdict},
            cls.PARAM_mod_path: mod_path,cls.PARAM_k:k,cls.PARAM_force_grey:force_grey, cls.PARAM_seed:seed,  "modcvt_dict": {}}}


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
