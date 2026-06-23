import time

import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
import collections
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.async_wrapper_agent import neko_infinite_workspace_shipper
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.log import info
import queue

# Will there be more ram usage due to queueing? yes. Are we doing it anyways? yes.
# bcs this separates data, meta, sinfo, lsct implemetation with loader.
# the thing will call things (asynchornized) in the following order
# diskldrs  -queue->  augments -queue-> meta -queue-> sinfo -queue-> lsct
# lsct is too diverse to be seen as pure data/meta, it can be pure random, making data wrt to meta, or making data wrt to real data.
from neko_sdk.log import fatal, warn, tree_view
import threading
class neko_320_training_loader_agent(neko_abstract_sync_agent):
    PARAM_diskloaders_acfg="diskloader_agents_cfg"; # DICT of ONE agent for all data, make it an awa if you load from different sources. Less io processes better io speed
    PARAM_augcol_acfgs="augcol_agents_acfg"; #DICT of "identical"augmenters with different seeds,  this will NOT seed and duplicate the agents with seeds, this will just launch them over queue, so you need to figure that yourself.
    PARAM_metas_acfg = "meta_agents_cfg"; # DICT of ONE  meta loading agent for all data. make it an awa if you load from different sources
    PARAM_sinfo_acfgs="sinfo_agents_cfg"; # DICT of "identical" sinfo loading agents, need to figure out its own augmentation, and collation. make them awas if you load from different sources
    PARAM_lsct_acfgs= "lsct_agents_cfg";
    PARAM_stageing_device="staging_device";

    QN_v2d=None;
    QN_2a = "->augcol";
    QN_2m = "->meta";
    QN_2s = "->sinfo";
    QN_2l = "->lsct";
    QN_2o= "->output";


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        pass;
    def set_venv(this):
        this.venv=neko_environment();
        # this.venv.replace_queue(this.QN_v2d);
        this.venv.replace_queue(this.QN_2a,maxsz=16);
        this.venv.replace_queue(this.QN_2m,maxsz=6);
        this.venv.replace_queue(this.QN_2s,maxsz=6);
        this.venv.replace_queue(this.QN_2l,maxsz=6);
        this.venv.replace_queue(this.QN_2o,maxsz=6);

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
            alist.append(awa.make(neko_infinite_workspace_shipper.get_agtcfg(iq, None, oq, None, acfg_dict[n])));
            nlist.append(n)
        return alist,nlist,eoq;

    def setup(this,params):

        this.allagts=[];
        this.allnames=[];
        # defaulting post queue gpu prefetch.  to turn that on you change "cpu" to "cuda:0"--- it does not work on berzelius
        this.staging_device=neko_get_arg(this.PARAM_stageing_device,params,"cuda");
        info("staging from", this.staging_device);
        lsctcfgs=neko_get_arg(this.PARAM_lsct_acfgs, params);
        curq=this.QN_v2d;

        this.diskloader,names,curq=this.arm_serial(neko_get_arg(this.PARAM_diskloaders_acfg, params),curq,this.QN_2a);
        this.allagts+=this.diskloader;
        this.allnames+=names;

        this.augcols,names,curq = this.arm_serial(neko_get_arg(this.PARAM_augcol_acfgs, params),curq, this.QN_2s);
        this.allagts+=this.augcols;
        this.allnames+=names;

        this.metas, names, curq = this.arm_serial(neko_get_arg(this.PARAM_metas_acfg, params), curq, this.QN_2m);
        this.allagts += this.metas;
        this.allnames += names;

        if(len(lsctcfgs)==0):
            sioq=this.QN_2o; # if no lscts are to be armed, directly output armed sideinfo to output queue
        else:
            sioq=this.QN_2l; # otherwise driect them to lsct agents.
        this.sinfos, names, curq = this.arm_serial(neko_get_arg(this.PARAM_sinfo_acfgs, params), curq, sioq);
        this.allagts += this.sinfos;
        this.allnames += names;

        this.lscts,names,curq = this.arm_serial(lsctcfgs,curq, this.QN_2o);
        this.allagts+=this.lscts;
        this.allnames+=names;
        if curq!=this.QN_2o:
            fatal("require rework to do nothing but ship");
        this.oq=curq;
        # time.sleep(3);
        this.launch();
        # time.sleep(3);
        this.buffer = queue.Queue(maxsize=4)  # Adjust depth based on VRAM


        # Start the prefetcher thread
        this.prefetch_thread = threading.Thread(target=this._prefetch_worker, daemon=True)
        this.prefetch_thread.start()
        pass;

    def _prefetch_worker(this):
        """Background thread to handle the slow queue reconstruction."""
        while True:
            # Reconstruction happens here in the background thread
            # info("trying to get data from ",this.oq);
            data = this.venv.deque(this.oq);
            # info("got data from ",this.oq);
            if(this.staging_device!="cpu"):
                data = {k:neko_workspace.move_to_device_recursive(data[k],this.staging_device) for k in data}
            # info("start putting data in ");
            this.buffer.put(data);
            # info("done putting data in ");


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        """Pops already reconstructed tensors from the local buffer."""
        # info("getting data out");
        data = this.buffer.get();
        # info("got data");
        with torch.no_grad():
            workspace.add_dict(data)

        return workspace, environment


    @classmethod
    def get_agtcfg(cls,
                   diskloaders_acfg, augcol_acfgs, metas_acfg, sinfo_acfgs, lsct_acfgs
                   ):
        assert (len(diskloaders_acfg)==1); # bcs random io is not making things faster, and can be shit inviting for hdds.
        assert (len(metas_acfg)==1);
        return {"agent": cls,
                "params": {cls.PARAM_augcol_acfgs: augcol_acfgs, cls.PARAM_diskloaders_acfg: diskloaders_acfg,
                           cls.PARAM_lsct_acfgs: lsct_acfgs, cls.PARAM_metas_acfg: metas_acfg,
                           cls.PARAM_sinfo_acfgs: sinfo_acfgs, "modcvt_dict": {}}}


