import cv2;

from neko_sdk.lmdb_wrappers.lmdb_wrapper import lmdb_wrapper;


class im_lmdb_wrapper(lmdb_wrapper):

    def add_data_utf_masked(this, image, gt, lang,gmask,lmask,maskedKey):

        imageKey = 'image-%09d'.encode() % (this.load+this.start);
        labelKey = 'label-%09d'.encode() % (this.load+this.start);
        langKey = 'lang-%09d'.encode() % (this.load+this.start);
        gmaskKey = 'gmask-%09d'.encode() % (this.load+this.start);
        lmaskKey = 'lmask-%09d'.encode() % (this.load+this.start);
        maskedKey = 'masked-%09d'.encode() % (this.load+this.start);

        this.txn.put(imageKey, cv2.imencode(".png", image)[1]);
        this.txn.put(labelKey, gt.encode());
        this.txn.put(langKey, lang.encode());
        this.txn.put()

        if (this.load % 500 == 0):
            this.txn.replace('num-samples'.encode(), str(this.load).encode());
            print("load:", this.load);
            this.txn.commit();
            del this.txn;
            this.txn = this.db.begin(write=True);
        this.load += 1;


    def add_data_utf(this,image,gt,lang):
        return this.adddata_kv({"image":image},{"label":gt,"lang":lang},{});
        #
        # imageKey = 'image-%09d'.encode() % this.load;
        # labelKey = 'label-%09d'.encode() % this.load;
        # langKey = 'lang-%09d'.encode() % this.load;
        #
        # this.txn.put(imageKey, cv2.imencode(".png", image)[1]);
        # this.txn.put(labelKey, gt.encode());
        # this.txn.put(langKey, lang.encode());
        # if (this.load % 500 == 0):
        #     this.txn.replace('num-samples'.encode(),str(this.load).encode());
        #     print("load:", this.load);
        #     this.txn.commit();
        #     del this.txn;
        #     this.txn = this.db.begin(write=True);
        # this.load += 1;
    def add_raw_data_utf(this,image,gt,lang):

        imageKey = 'image-%09d'.encode() % this.load+this.start;
        labelKey = 'label-%09d'.encode() % this.load+this.start;
        langKey = 'lang-%09d'.encode() % this.load+this.start;

        this.txn.put(imageKey, image);
        this.txn.put(labelKey, gt.encode());
        this.txn.put(langKey, lang.encode());
        if (this.load % 500 == 0):
            this.txn.replace('num-samples'.encode(),str(this.load).encode());
            print("load:", this.load);
            this.txn.commit();
            del this.txn;
            this.txn = this.db.begin(write=True);
        this.load += 1;
