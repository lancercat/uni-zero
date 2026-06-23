import sys

import lmdb
import six
from PIL import Image

from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_abstract_data_source_NG
from neko_sdk.cfgtool.argsparse import neko_get_arg

from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from neko_sdk.log import warn
class neko_lmdb_holder(neko_abstract_data_source_NG):
    PARAM_root="root"
    PARAM_vert_to_hori="vert_to_hori"
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
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root[i]))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            this.nSamples += nSamples
        this.env = env;
        this.reset_txn();

    def __len__(this):
        return this.nSamples

    def setup(this, para):
        this.disarm();
        this.arm_lmdb(para[this.PARAM_root]);
        this.vert_to_hori=neko_get_arg(this.PARAM_vert_to_hori,para,-100000);
        this.vert_to_hori = -100000;
        this.init_etc(para);

    def fetch_core(this, index):

        img_key = 'image-%09d' % index;
        label_key = 'label-%09d' % index;
        label = str(this.txn.get(label_key.encode()).decode('utf-8'));
        imgbuf = this.txn.get(img_key.encode())
        buf = six.BytesIO();
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf);
        if (img.width / img.height < this.vert_to_hori):
            img = img.transpose(Image.ROTATE_90);
        if(img.mode != "RGB"): # well I hate PIL....
            img= img.convert("RGB");
        if(len(label)==0):
            warn("please don't feed empty image to the model"+img_key);
            return None;

        ret = {
            RN.IMAGE: img,
            RN.LABEL: label,
        };
        return ret;

    def fetch_item(this, descp):
        index = (descp["id"]);
        ret = None;
        try:
            ret = this.fetch_core(index);
        except:
            try:
                this.reset_txn();
                ret = this.fetch_core(index);
                print("bad_txn", "resetted");
            except:
                warn('Corrupted image for %d' % index);
        return ret;

    def all_valid_indexes(this):
        return [{"id": i} for i in range(this.nSamples)];
