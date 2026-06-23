from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.async_wrapper_agent import neko_infinite_workspace_shipper
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.log import info


# Will there be more ram usage due to queueing? yes. Are we doing it anyways? yes.
# bcs this separates data, meta, sinfo, lsct implemetation with loader.
# the thing will call things (asynchornized) in the following order
# diskldrs -queue-> lscts -queue->  augments -queue-> meta -queue-> sinfo
class neko_training_loader_agent(neko_abstract_sync_agent):
    PARAM_diskloaders_acfg="diskloader_agents_cfg"; # DICT of ONE agent for all data, make it an awa if you load from different sources. Less io processes better io speed
    PARAM_augcol_acfgs="augcol_agents_acfg"; #DICT of "identical"augmenters with different seeds,  this will NOT seed and duplicate the agents with seeds, this will just launch them over queue, so you need to figure that yourself.
    PARAM_lsct_acfgs = "lsct_agents_cfg"; # DICT of "identical" lsct agents, need to figure out its own augmentation, and collation. make them awas if you load from different sources
    PARAM_metas_acfg = "meta_agents_cfg"; # DICT of ONE  meta loading agent for all data. make it an awa if you load from different sources
    PARAM_sinfo_acfgs="sinfo_agents_cfg"; # DICT of "identical" sinfo loading agents, need to figure out its own augmentation, and collation. make them awas if you load from different sources

    QN_v2d=None;
    QN_d2a = "->augcol";
    QN_a2l = "->lsct";
    QN_l2m = "->meta";
    QN_m2s = "->sinfo";
    QN_s2o = "->output";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        pass;
    def set_venv(this):
        this.venv=neko_environment();
        # this.venv.replace_queue(this.QN_v2d);
        this.venv.replace_queue(this.QN_d2a,maxsz=16);
        this.venv.replace_queue(this.QN_a2l,maxsz=6);
        this.venv.replace_queue(this.QN_l2m,maxsz=6);
        this.venv.replace_queue(this.QN_m2s,maxsz=6);

    def launch(this):
        this.set_venv();
        for a,n in zip(this.allagts,this.allnames):
            info("starting",n);
            a.start(this.venv, "forkserver");
            info("started",n);
    def arm_serial(this,acfg_dict,iq,oq):
        alist=[];
        nlist=[];
        if(len(acfg_dict)==0):
            eoq=iq;
        else:
            eoq=oq;
        for n in acfg_dict:
            alist.append(awa.make(neko_infinite_workspace_shipper.get_agtcfg(iq, [], oq, None, acfg_dict[n])));
            nlist.append(n)
        return alist,nlist,eoq;

    def set_etc(this, params):
        this.allagts=[];
        this.allnames=[];
        curq=this.QN_v2d;

        this.diskloader,names,curq=this.arm_serial(neko_get_arg(this.PARAM_diskloaders_acfg, params),curq,this.QN_d2a);
        this.allagts+=this.diskloader;
        this.allnames+=names;

        this.augcols,names,curq = this.arm_serial(neko_get_arg(this.PARAM_augcol_acfgs, params),curq, this.QN_a2l);
        this.allagts+=this.augcols;
        this.allnames+=names;

        this.lscts,names,curq = this.arm_serial(neko_get_arg(this.PARAM_lsct_acfgs, params),curq, this.QN_l2m);
        this.allagts+=this.lscts;
        this.allnames+=names;

        this.metas, names, curq = this.arm_serial(neko_get_arg(this.PARAM_metas_acfg, params), curq, this.QN_m2s);
        this.allagts += this.metas;
        this.allnames += names;

        this.sinfos, names, curq = this.arm_serial(neko_get_arg(this.PARAM_sinfo_acfgs, params), curq, this.QN_s2o);
        this.allagts += this.sinfos;
        this.allnames += names;

        this.launch();
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   diskloaders_acfg, augcol_acfgs, lsct_acfgs, metas_acfg, sinfo_acfgs
                   ):
        assert (len(diskloaders_acfg)==1);
        assert (len(metas_acfg)==1);
        return {"agent": cls,
                "params": {cls.PARAM_augcol_acfgs: augcol_acfgs, cls.PARAM_diskloaders_acfg: diskloaders_acfg,
                           cls.PARAM_lsct_acfgs: lsct_acfgs, cls.PARAM_metas_acfg: metas_acfg,
                           cls.PARAM_sinfo_acfgs: sinfo_acfgs, "modcvt_dict": {}}}



if __name__ == '__main__':
    neko_training_loader_agent.print_default_setup_scripts();
