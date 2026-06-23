
### data loading agent is logically single threaded
### But heck, you can parallel the processing yourself.
### Async part is done within batch.
### The bottom line is, to load and to augment in different modules, sync or not.
### Also, you can always have move data to CUDA using another async module.
### In all, it looks like this:
### [dataloader]<data and augment recipe>[Augmenter(MT)]<augmented data>[Cuda mover]
### If you want to have learnable augmenter, just set [Augmenter(MT)] doing nothing....
## Why such a hassle? bcs we want to make LSCT and augmentation determinstic!

class neko_abstract_data_source_NG:
    def setup(this,para):
        pass;

    ### The descp is a dict, so that is why we cannot use the dataset class here.
    def fetch_item(this, descp):
        pass;
    def disarm(this):
        this.nSamples = 0;

    def __init__(this,para):
        this.nSamples=0;
        this.setup(para);
    def all_valid_indexes(this):
        return [];




### Now we do not have the necessity for multi-lmdb-holder, not on API level.
### We can just abstract one mixer class and let the user to mux hierarchically
class neko_named_multi_source_holder(neko_abstract_data_source_NG):
    PARAM_sources="sources";
    PARAM_sourced="sourced";
    def init_etc(this,para):
        pass;
    def disarm(this):
        this.sourced = {};
        this.sources=[];
        this.nSamples=0
    def setup(this,para):
        this.disarm();
        this.sources=para[this.PARAM_sources];
        for k in para[this.PARAM_sources]:
            this.sourced[k]=para[this.PARAM_sourced][k];
            this.nSamples+=this.sourced[k].nSamples;

    def fetch_item(this, descp):
        if("dskey" not in descp):
            fromwhich=descp["dsid"]%len(this.sources);
            return this.sourced[this.sources[fromwhich]].fetch_item(descp["descp"]);
        elif(descp["dskey"] in this.sourced):
            return this.sourced[this.sources[descp["dskey"]]].fetch_item(descp["descp"]);
        else:
            print("invalid data index", descp);
    def reset_txn(this,descp):
        fromwhich = descp["dsid"] % len(this.sources);
        this.sourced[this.sources[fromwhich]].reset_txn();

    def all_valid_indexes(this):
        idxs=[];
        for i in range(len(this.sources)):
            idxs+=[{"dsid":i,"descp":_} for _ in this.sourced[this.sources[i]].all_valid_indexes()];
        return idxs;

