
import os

from neko_sdk.ocr_modules.lmdbcvt.hwdbcvt import make_hwdb
from osocr_tasks.tasksg1.dscs import scanfolder_and_add_pt
from osocr_tasks.tasksg1.hwdb_fslchr.split import t1shuflist
from osocr_tasks.tasksg1.taskmaker import build_training_dataset_by_label, build_testing_dataset_by_label, \
    split_labelset;


def splithwdbtr(trfulldbdst,tefulldbdst,name,
                dstroot,train_cnt,evalcnt,valcnt):
    # os.makedirs(dstroot,exist_ok=True);
    train,eval,val=split_labelset(t1shuflist,train_cnt,evalcnt,valcnt);
    build_training_dataset_by_label([trfulldbdst,tefulldbdst], dstroot,train, ["image"], ["label", "lang", "wrid"], "label",name);
    # torch.save([train,val,eval],os.path.join(dstroot,"split.pt"));
def splithwdb_test(trfulldbdst, tefulldbdst, compdbdst,name,
                   dstroot, train_cnt, evalcnt, valcnt):
    dstroot=os.path.join(dstroot,name);
    train,eval,val=split_labelset(t1shuflist,train_cnt,evalcnt,valcnt);

    build_testing_dataset_by_label([tefulldbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuws_eval"+name,"cuws_val"+name);
    build_testing_dataset_by_label([compdbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuwu_eval"+name,"cuwu_val"+name);


    # build_testing_dataset_by_label([compdbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuwu_eval","cuwu_val");
    # build_training_dataset_by_label([compdbdst], dstroot, train, ["image"], ["label", "lang", "wrid"], "label", "cswu_eval");


def buildhwdb(trfulldb,tefulldb,compdb,droot):
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_20",droot,2000,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_15",droot,
                1500,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_10",droot,
                1000,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_5",droot,
                500,1000,100);

    splithwdb_test(trfulldb, tefulldb, compdb, "hwdbfsl_10_1",
                   droot, 2000, 1000, 100);




# it seems to be a tradition using 1.0-1.2 testing set for training.
# and as you can see, the validation set and evaluation set are always the same
# so we manually removed redundant
if __name__ == '__main__':
    import sys
    if (len(sys.argv)>1):
        ROOT = sys.argv[1];
        CROOT = sys.argv[2];
        DROOT = sys.argv[3];
    else:
        ROOT="/run/media/lasercat/writebuffer/deploy/"
        CROOT = "/run/media/lasercat/writebuffer/cachededlmdbs/"
        DROOT = "/run/media/lasercat/cache2/"

    trgnt=ROOT+"hwdb/train/"
    tegnt=ROOT+"hwdb/test/"
    i13gnt=ROOT+"hwdb/comp/"

    trfulldbdst = CROOT+"HWDB/hwdbtr";
    tefulldbdst = CROOT+"HWDB/hwdbte";
    i13fulldbdst= CROOT+"HWDB/hwdbco"
    fsltsks=DROOT+"HWDB/pami_ch_fsl_hwdb";
    fnts=[ROOT+"fonts/NotoSansCJK-Regular.ttc"]
    make_hwdb(trgnt, trfulldbdst);
    make_hwdb(tegnt, tefulldbdst);
    make_hwdb(i13gnt, i13fulldbdst);
    os.makedirs(fsltsks, exist_ok=True);
    buildhwdb(trfulldbdst,tefulldbdst,i13fulldbdst,fsltsks);
    scanfolder_and_add_pt(fsltsks,fnts,set(),set());