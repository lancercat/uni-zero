import cv2;

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper


class neko_fsl_lmdb_wrapper(im_lmdb_wrapper):
    def add_data_fsl(this,image,gt):
        imageKey = 'image-%09d'.encode() % this.load;
        labelKey = 'label-%09d'.encode() % this.load;
        this.txn.put(imageKey, cv2.imencode(".png", image)[1]);
        this.txn.put(labelKey, gt.encode());
        if (this.load % 500 == 0):
            this.txn.replace('num-samples'.encode(),str(this.load).encode());
            print("load:", this.load);
            this.txn.commit();
            del this.txn;
            this.txn = this.db.begin(write=True);
        this.load += 1;
