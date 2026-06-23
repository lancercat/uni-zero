import sys

import lmdb
import six
from PIL import Image

from neko_sdk.neko_framework_NG.data.neko_data_source_NG import neko_abstract_data_source_NG
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.log import warn,fatal

class neko_abstract_lmdb_loader(neko_abstract_data_source_NG):
    PARAM_root="root"
    def __getstate__(this):
        state = this.__dict__
        state["env"] = None;
        state["txn"] = None;
        return state

    def __setstate__(this, state):
        this.__dict__ = state
        env = lmdb.open(
            this.lmdb_root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False);
        this.env = env;
        this.reset_txn();

    def reset_txn(this):
        this.txn = this.env.begin(write=False)

    def init_etc(this, para):
        pass;

    def disarm(this):
        this.root = None;
        this.envs = None;
        this.nSamples = 0;

    def arm_lmdb(this, root):

        this.lmdb_root = root;
        env = lmdb.open(
            root,
            readonly=True,
            lock=False,
            readahead=False)
        if not env:
            fatal('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            this.nSamples += nSamples
        this.env = env;
        this.reset_txn();

    def __len__(this):
        return this.nSamples
    def fetch_core(this,descp):
        fatal("notimpl");

    def fetch_item(this, descp):
        ret = None;
        try:
            ret = this.fetch_core(descp);
        except:
            try:
                this.reset_txn();
                ret = this.fetch_core(descp);
                print("bad_txn", "resetted");
            except:
                warn('Corrupted image for',descp);
        return ret;

    def setup(this, para):
        this.disarm();
        this.root=para[this.PARAM_root];
        this.arm_lmdb(para[this.PARAM_root]);
        this.init_etc(para);
    def fetch_img(this,img_key):
        imgbuf = this.txn.get(img_key.encode())
        buf = six.BytesIO();
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf);
        if (img.mode != "RGB"):  # well I hate PIL....
            img = img.convert("RGB");
        return img;
    def fetch_text(this,text_key):
        return str(this.txn.get(text_key.encode()).decode('utf-8'));
    def fetch_blob(this,key):
        return this.txn.get(key.encode());
    def fetch_blob_by_index(this,key,index):
        skey = f"{key}-{index:09d}"
        return this.fetch_blob(skey);