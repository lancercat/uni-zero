import pickle;

from neko_sdk.lmdb_wrappers.lmdb_wrapper import lmdb_wrapper;


class corpus_lmdb_wrapper(lmdb_wrapper):
    def __init__(this,dbpath):
        super(corpus_lmdb_wrapper,this).__init__(dbpath);

    def add_data_utf(this,content,compatible_list):
        contentKey = 'content-%09d'.encode() % this.load;

        this.txn.put(contentKey,content.encode() );
        if(compatible_list is not None):
            compatKey = 'compatible-%09d'.encode() % this.load;
            this.txn.put(compatKey, pickle.dumps(compatible_list));
        if (this.load % 500 == 0):
            this.txn.replace('num-samples'.encode(),str(this.load).encode());
            print("load:", this.load);
            this.txn.commit();
            del this.txn;
            this.txn = this.db.begin(write=True);
        this.load += 1;
