import os
from struct import pack

import numpy as np
import torch

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper;


class generic_pot_lmdb_wrapper(im_lmdb_wrapper):

    def end_this(this):
        this.txn.replace('num-samples'.encode(), str(this.load).encode());
        this.txn.commit();
        this.db.close();
        torch.save({"labels": this.labels, "writers": this.writers}, os.path.join(this.root, "meta.pt"));
    def add_char_utf(this,image,tagcode,wrid):
        char = pack('>H', tagcode).decode('gb2312');
        imagefn = "label_%05d_%s" % (tagcode, wrid);
        this.writers.add(wrid);
        this.labels[char]=tagcode;
        this.adddata_kv({"image":image},{"label":char,"lang":"Chinese","wid":wrid},{})

    def pil2cv_c1(this,pil_image):
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image.copy()
        return open_cv_image;
