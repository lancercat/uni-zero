import copy
import tqdm

from osocrNG.data_utils.data_agents.sync_loaders.abstract_sync_im_text_loader import neko_abstract_im_text_sync_random_multidb
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from osocrNG.data_utils.raw_names import basic_data_info_mk2 as BIN2

from neko_sdk.log import fatal,warn
import pickle
from neko_sdk.log import info

# from 32x data loader only answers to training range. All data fused to one single data pile.
class neko_sync_balance_fetching_and_mixing_agent(neko_abstract_im_text_sync_random_multidb):
    PARAM_ancidx_path="ancidx_path";
    PARAM_anchor_cfg="anchor_cfg";
    PARAM_ancidx="ancidx_instance";
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        imlst=[];
        utflst=[];
        dscplist=[];

        for k in this.anchor_names:
            idxs=this.ancidx[k];
            bs = this.batch_sizes[k];
            if(len(idxs)==0):
                continue;
            ti=this.rng.choices(idxs,k=bs)
            for idx in ti:
                data=this.holder.fetch_item(idx);
                if (data is not None):
                    imlst.append(data[BIN2.IMAGE]);
                    utflst.append(data[BIN2.LABEL]);
                    dscplist.append(copy.copy(idx));
        if (this.force_grey):
            imlst=this.imlst_as_grey(imlst);
        workspace.add(this.images,imlst);
        workspace.add(this.text,utflst);
        workspace.add(this.dscps,dscplist);



    def make_anc_idx(this):
        this.ancidx={};
        for k in this.anchor_names:
            this.ancidx[k]=[];
        for idx in tqdm.tqdm(this.holder.all_valid_indexes()):
            if(idx["descp"]["id"]%100==9):
                this.holder.reset_txn(idx)
            data=this.holder.fetch_item(idx);
            try:
                data[RN.IMAGE].tobytes();
            except:
                warn("hidden corrupted sample");
                continue

            if(data is not None):
                ratio = data[RN.IMAGE].width / data[RN.IMAGE].height;
                txt = data[RN.LABEL];
                putcnt = 0;
                for i in range(len(this.ratio_anchors)):
                    if (len(txt) > this.maxT[i] or len(txt) < this.minT[i]):
                        continue;
                    if (ratio < this.ratio_anchors[i][0] or ratio> this.ratio_anchors[i][1]):
                        continue;
                    this.ancidx[this.anchor_names[i]].append(idx);
                    putcnt+=1;
                if (putcnt == 0):
                    print("we get an orphan:", idx, "txt:", txt);


    def setmeta(this,param):
        this.ancidx_path=param[this.PARAM_ancidx_path];
        this.anchor_names = param[this.PARAM_anchor_cfg]["names"];

        this.ratio_anchors = [
            param[this.PARAM_anchor_cfg][k]["training_ar_range"] for k  in this.anchor_names];
        this.maxT=[
            param[this.PARAM_anchor_cfg][k]["maxT"] for k in this.anchor_names];
        this.minT = [
            param[this.PARAM_anchor_cfg][k]["minT"] for k in this.anchor_names];
        this.batch_sizes = {};
        for k in this.anchor_names:
            this.batch_sizes[k] = param[this.PARAM_anchor_cfg][k]["batch_size"];

    def set_etc(this,param):
        super().set_etc(param)
        this.setmeta(param);
        if (this.PARAM_ancidx not in param):
            try:
                info("loading anchor index",this.ancidx_path);
                with open(this.ancidx_path,"rb") as fp:
                    this.ancidx=pickle.load(fp);
                info("anchor index loaded");

            except:
                this.make_anc_idx();
                with open(this.ancidx_path,"wb+") as fp:
                    pickle.dump(this.ancidx,fp);
            info("anchor index loaded and secured");
        else:
            this.ancidx=param[this.PARAM_ancidx];

    @classmethod
    def get_agtcfg(cls,
                   dscps,images, text,
                   anchor_cfg, ancidx_path, names, roots, seed,force_grey=False
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.OUTPUT_dscps:dscps, cls.OUTPUT_images: images, cls.OUTPUT_text: text},
                                         cls.PARAM_anchor_cfg: anchor_cfg, cls.PARAM_ancidx_path: ancidx_path,
                                         cls.PARAM_batch_size: -9, cls.PARAM_names: names, cls.PARAM_force_grey:force_grey,
                                         cls.PARAM_roots: roots, cls.PARAM_seed: seed, "modcvt_dict": {}}}



# from 32x data loader only answers to training range. All data fused to one single data pile.
class neko_im_text_sync_balance_fetching_and_mixing_mk2_agent(neko_sync_balance_fetching_and_mixing_agent):


    def setmeta(this,param):
        this.ancidx_path=param[this.PARAM_ancidx_path];
        this.anchor_names = param[this.PARAM_anchor_cfg]["names"];

        this.ratio_anchors = [
            param[this.PARAM_anchor_cfg][k]["training_ar_range"] for k  in this.anchor_names];
        this.maxT=[
           9999 for k in this.anchor_names];
        this.minT = [
            -9999 for k in this.anchor_names];
        this.batch_sizes = {};
        for k in this.anchor_names:
            this.batch_sizes[k] = param[this.PARAM_anchor_cfg][k]["batch_size"];



if __name__ == '__main__':
    # neko_balance_fetching_and_mixing_agent.print_default_setup_scripts()
    from neko_2024_NGNW.nets_v6.anchors import get_wna_v6_dcfg_1h1v1r_2_05_smol as trdcfg
    agte=neko_sync_balance_fetching_and_mixing_agent
    acfg=agte.get_agtcfg(
        "image","text",trdcfg(24),"/home/lasercat/writebuffer/tmp/tmpidx.pt",["rctwtrdb_seen"],
            ["/home/lasercat/ssddata/rctwtrdb_seen"]
        ,9
    );
    a=agte(acfg["params"]);
    ws=neko_workspace();
    e=neko_environment();
    a.take_action(ws,e);
    pass;

