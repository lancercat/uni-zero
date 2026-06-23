import sys

import lmdb
import six
from PIL import Image

from neko_sdk.neko_framework_NG.data.abstract_lmdb_loader import neko_abstract_lmdb_loader
from neko_sdk.cfgtool.argsparse import neko_get_arg

from osocrNG.data_utils.raw_names import basic_data_info as RN
from neko_sdk.log import warn
class neko_im_label_lmdb_holder(neko_abstract_lmdb_loader):
    PARAM_vert_to_hori="vert_to_hori"
    def get_img_key(this,descp):
        index = (descp["id"]);
        img_key = 'image-%09d' % index;
        return img_key;

    def fetch_img_label(this,index):
        img_key = 'image-%09d' % index;
        label_key = 'label-%09d' % index;
        label = this.fetch_text(label_key);
        img = this.fetch_img(img_key);
        return img,label;

    def init_etc(this, para):
        this.vert_to_hori=neko_get_arg(this.PARAM_vert_to_hori,para,-100000);
        this.vert_to_hori = -100000;

    def fetch_core(this,descp):
        index = descp["id"];
        img,label=this.fetch_img_label(index);

        if (img.width / img.height < this.vert_to_hori):
            img = img.transpose(Image.ROTATE_90);
        if (len(label) == 0):
            img_key = 'image-%09d' % index;
            warn("please don't feed empty image to the model" + img_key);
            return None;
        ret = {
            RN.IMAGE: img,
            RN.LABEL: label,
        };
        return ret;

        return this.fetch_img_label(index);

    def all_valid_indexes(this):
        return [{"id": i} for i in range(this.nSamples)];
