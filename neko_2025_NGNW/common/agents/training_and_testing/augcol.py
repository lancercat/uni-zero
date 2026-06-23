
from neko_2025_NGNW.common.object_32x_presets.deprecated import neko_aug_swarm_sync_factory_abi

from neko_sdk.neko_framework_NG.UAE.async_wrapper_agent import neko_infinite_workspace_shipper

from neko_2025_NGNW.common.object_32x_presets.var_names import P

# activelearning gonna wait after we have blackwell together with better io. Prefetching is still the life line here. So we cannot afford gradient


from osocrNG.data_utils.common_data_presets_mk5.data.deprecated.list_im_collators import neko_im_text_collate_factory



class augment_col:
    COL = neko_im_text_collate_factory;
    AU=neko_aug_swarm_sync_factory_abi;
    def __init__(this, data_template):
        this.data_template=data_template;

    def mk_sync_augment_agent(this,data_streams, lsct_extensions, meta_streams, seed, data_root):
        ks=list(data_streams.keys());
        nlist = ["-".join(ks)];
        df =this.AU (data_root, seed);
        for k in data_streams:
            df.add_data_pipeline(k, this.data_template,data_streams[k]);
        for exk in lsct_extensions:
            df.add_shadow_lsct_to_datastream(exk,this.data_template);
        acfg = df.get_swarm_agent();
        return acfg, nlist;
    def mk_async_augment_agent(this,data_streams, lsct_extensions, meta_streams, seed, data_root,iqn,oqn):
        acfg, nlist = this.mk_sync_augment_agent(data_streams, lsct_extensions, meta_streams, seed, data_root);
        hcfg = neko_infinite_workspace_shipper.get_agtcfg(iqn, None, oqn, None, acfg);
        return hcfg, nlist

    def mk_async_augment_agents(this,data_streams, lsct_extensions, meta_streams, seeds, data_root,iqn,oqn):
        ahcfg = [];
        nlist = [];
        for seed in seeds:
            hcfg, nlist_ = this.mk_async_augment_agent(data_streams, lsct_extensions, meta_streams, seed, data_root,iqn, oqn);
            nlist+=[P(n,str(seed)) for n in nlist_];
            ahcfg.append(hcfg);
        return ahcfg, nlist;


