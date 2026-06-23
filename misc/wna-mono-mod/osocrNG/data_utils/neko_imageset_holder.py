import cv2
import numpy as np
from PIL import Image

from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_abstract_data_source_NG


class neko_image_holder(neko_abstract_data_source_NG):
    def disarm(this):
        this.nSamples = 0;
    def __len__(this):
        return this.nSamples;

    def setup(this, para):
        this.disarm();
        this.files = (para["files"]);
        if ("gts" in para):
            this.gts = para["gts"];
        else:
            this.gts=None;
        this.nSamples = len(this.files);
        this.vert_to_hori = -100000;
        if ("vert_to_hori" in para):
            this.vert_to_hori = para["vert_to_hori"]

    def fetch_core(this, index):
        with open(this.files[index], "rb") as fp:
            img = Image.open(fp).convert("RGB");
        if (img.width / img.height < this.vert_to_hori):
            img = img.transpose(Image.ROTATE_90);
        ret = {
            "image":img,
        };
        return ret;

    def fetch_item(this, descp):
        # zero start here.
        index = (descp["id"]-1);
        ret = None;
        try:
            ret = this.fetch_core(index);
        except:
            warn('Corrupted image for %d' % index);
        return ret;

    def all_valid_indexes(this):
        return this.files;

