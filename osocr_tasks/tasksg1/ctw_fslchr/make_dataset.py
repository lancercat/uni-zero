import sys

from neko_sdk.ocr_modules.lmdbcvt.ctwchcvt import make_ctwch
from osocr_tasks.tasksg1.ctw_fslchr.split import ctwshuflist
from osocr_tasks.tasksg1.dscs import scanfolder_and_add_pt;
from osocr_tasks.tasksg1.taskmaker import build_training_dataset_by_label, build_testing_dataset_by_label, \
    split_labelset;


def splitctwfsl_tr(srcdb,dstroot,train_cnt,name,evalcnt=500,valcnt=100):
    train,eval,val=split_labelset(ctwshuflist,train_cnt,evalcnt,valcnt);
    build_training_dataset_by_label(srcdb, dstroot,train, ["image"], ["label", "lang", "attr"], "label",name=name);

def splitctwfsl_te(srcdb, dstroot, train_cnt, evalname,valname, evalcnt=500, valcnt=100):
    train,eval,val=split_labelset(ctwshuflist,train_cnt,evalcnt,valcnt);
    build_testing_dataset_by_label(srcdb, dstroot, val, eval, ["image"], ["label", "lang", "attr"], "label",evalname=evalname,valname=valname);

def buildctws(src,droot,fntpath):
    splitctwfsl_tr([src],droot,500,"ctwfsl5_train",500,100);
    splitctwfsl_tr([src],droot,1000,"ctwfsl10_train",500,100);
    splitctwfsl_tr([src],droot,1500,"ctwfsl15_train",500,100);
    splitctwfsl_tr([src],droot,2000,"ctwfsl20_train",500,100);
    splitctwfsl_te([src], droot, 2000, "ctwfsl_5_1eval","ctwfsl_5_1val", 500, 100);




if __name__ == '__main__':
    if (len(sys.argv)>1):
        ROOT = sys.argv[1];
        CROOT = sys.argv[2];
        DROOT = sys.argv[3];
    else:
        ROOT="/run/media/lasercat/writebuffer/deploy/"
        CROOT = "/run/media/lasercat/writebuffer/cachededlmdbs/"
        DROOT = "/run/media/lasercat/cache2/"
    trgtpath = ROOT+"ctw/gtar/train.jsonl";
    trjpgpath = ROOT+"ctw/itar";
    chfulldbdst = CROOT+"ctwch";
    fsltsks=DROOT+"ctwch";
    fntpath=ROOT+"fonts/NotoSansCJK-Regular.ttc"
    make_ctwch(trgtpath, trjpgpath,chfulldbdst);
    buildctws(chfulldbdst, fsltsks,fntpath);
    scanfolder_and_add_pt(fsltsks,[fntpath],set(),set());
