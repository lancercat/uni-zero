import glob
import os.path

from osocr_tasks.lmdbcvt.imgt import imgt_lmdbexport
import cv2

class crop_word_export(imgt_lmdbexport):
    def handle_file(this,im,gt):
        im=cv2.imread(im);
        with open(gt,"r") as fp:
            for content in fp:
                this.db.add_data_utf(im, content, "Unspecified");
    @classmethod
    def mkfolder(cls,src,dst):
        e=cls(dst);
        keys=[os.path.basename(_).replace(".jpg","") for _ in glob.glob(os.path.join(src,"*.jpg"))];
        impaths = [];
        gtpaths = [];
        for k in keys:
            imname = os.path.join(src, k + ".jpg");
            gtname = os.path.join(src, "gt_" + k + ".txt");
            gtpaths.append(gtname);
            impaths.append(imname);
        e.add_ds_by_pairs(impaths,gtpaths);
        e.end_adding();
    @classmethod
    def mk_combined_folder(cls,srcs,dst):
        e=cls(dst);
        for src in srcs:
            keys=[os.path.basename(_).replace(".jpg","") for _ in glob.glob(os.path.join(src,"*.jpg"))];
            impaths = [];
            gtpaths = [];
            for k in keys:
                imname = os.path.join(src, k + ".jpg");
                gtname = os.path.join(src, "gt_" + k + ".txt");
                gtpaths.append(gtname);
                impaths.append(imname);
            e.add_ds_by_pairs(impaths,gtpaths);
        e.end_adding();
if __name__ == '__main__':
    crop_word_export.mkfolder("/run/media/lasercat/writebuffer/synth_3k_3755_random_maxl_8/","/run/media/lasercat/writebuffer/synth_3k_3755_random_maxl_8_lmdb/");
    crop_word_export.mkfolder("/run/media/lasercat/writebuffer/synth_3k_3755_random_maxl_8_alter/","/run/media/lasercat/writebuffer/synth_3k_3755_random_maxl_8_alter_lmdb/");